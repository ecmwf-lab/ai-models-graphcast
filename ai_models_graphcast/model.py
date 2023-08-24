# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import dataclasses
import datetime
import functools
import logging
import os
from functools import cached_property

import climetlab as cml
import haiku as hk
import jax
import numpy as np
import xarray
import xarray as xr
from ai_models.model import Model
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
)

LOG = logging.getLogger(__name__)


class GraphcastModel(Model):
    download_url = "https://storage.googleapis.com/dm_graphcast/{file}"
    expver = "dmgc"

    # Download
    download_files = [
        (
            "params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 -"
            " pressure levels 13 - mesh 2to6 - precipitation output only.npz"
        ),
        "stats/diffs_stddev_by_level.nc",
        "stats/mean_by_level.nc",
        "stats/stddev_by_level.nc",
    ]

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]

    param_sfc = [
        "lsm",
        "2t",
        "msl",
        "10u",
        "10v",
        "tp",
        "z",
    ]

    param_level_pl = (
        ["t", "z", "u", "v", "w", "q"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    forcings = [
        "toa_incident_solar_radiation",  # should be tisr
        "sin_julian_day",
        "cos_julian_day",
        "sin_local_time",
        "cos_local_time",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hour_steps = 6
        self.lagged = [-6, 0]
        self.params = None
        self.ordering = self.param_sfc + [
            f"{param}{level}"
            for param in self.param_level_pl[0]
            for level in self.param_level_pl[1]
        ]

    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def _with_configs(self, fn):
        return functools.partial(
            fn, model_config=self.model_config, task_config=self.task_config
        )

    # Always pass params and state, so the usage below are simpler
    def _with_params(self, fn):
        return functools.partial(fn, params=self.params, state=self.state)

    # Our models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by our rollout code, and generally simpler.
    @staticmethod
    def _drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    def load_model(self):
        with self.timer(f"Loading {self.download_files[0]}"):

            def get_path(filename):
                return os.path.join(self.assets, filename)

            diffs_stddev_by_level = xarray.load_dataset(
                get_path(self.download_files[1])
            ).compute()
            mean_by_level = xarray.load_dataset(
                get_path(self.download_files[2])
            ).compute()
            stddev_by_level = xarray.load_dataset(
                get_path(self.download_files[3])
            ).compute()

            def construct_wrapped_graphcast(model_config, task_config):
                """Constructs and wraps the GraphCast Predictor."""
                # Deeper one-step predictor.
                predictor = graphcast.GraphCast(model_config, task_config)

                # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
                # from/to float32 to/from BFloat16.
                predictor = casting.Bfloat16Cast(predictor)

                # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
                # BFloat16 happens after applying normalization to the inputs/targets.
                predictor = normalization.InputsAndResiduals(
                    predictor,
                    diffs_stddev_by_level=diffs_stddev_by_level,
                    mean_by_level=mean_by_level,
                    stddev_by_level=stddev_by_level,
                )

                # Wraps everything so the one-step model can produce trajectories.
                predictor = autoregressive.Predictor(
                    predictor, gradient_checkpointing=True
                )
                return predictor

            @hk.transform_with_state
            def run_forward(
                model_config, task_config, inputs, targets_template, forcings
            ):
                predictor = construct_wrapped_graphcast(model_config, task_config)
                return predictor(
                    inputs, targets_template=targets_template, forcings=forcings
                )

            with open(get_path(self.download_files[0]), "rb") as f:
                self.ckpt = checkpoint.load(f, graphcast.CheckPoint)
                self.params = self.ckpt.params
                self.state = {}

                self.model_config = self.ckpt.model_config
                self.task_config = self.ckpt.task_config

                LOG.info("Model description: %s", self.ckpt.description)
                LOG.info("Model license: %s", self.ckpt.license)

            init_jitted = jax.jit(self._with_configs(run_forward.init))
            self.model = self._drop_state(
                self._with_params(jax.jit(self._with_configs(run_forward.apply)))
            )

    @cached_property
    def start_date(self) -> "datetime":
        return self.all_fields.order_by(valid_datetime="descending")[0].datetime

    def cml_variables(self, date: "datetime") -> "torch.Tensor":
        """Generate variables from climetlabs

        Args:
            date (datetime): Datetime of current time step in forecast
            params (List[str]): Parameters to calculate as constants

        Returns:
            torch.Tensor: Tensor with constants
        """
        ds = cml.load_source(
            "constants", self.all_fields, date=date, param=self.forcings
        )

        return (
            ds.order_by(param=self.forcings, valid_datetime="ascending")
            .to_numpy()
            .reshape(len(self.forcings), len(date), 721, 1440)
        )

    def create_graphcast_inputs(self):
        # Create Input dataset
        self.sfc_names = {
            "t2m": "2m_temperature",
            "msl": "mean_sea_level_pressure",
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "tp": "total_precipitation_6hr",
            "z": "geopotential_at_surface",
            "lsm": "land_sea_mask",
            "latitude": "lat",
            "longitude": "lon",
            "step": "batch",
            "valid_time": "datetime",
        }
        self.pl_names = {
            "t": "temperature",
            "z": "geopotential",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "w": "vertical_velocity",
            "q": "specific_humidity",
            "isobaricInhPa": "level",
            "latitude": "lat",
            "longitude": "lon",
            "step": "batch",
            "valid_time": "datetime",
        }
        self.sfc_fields = (
            self.fields_sfc.to_xarray().rename(self.sfc_names).isel(number=0, surface=0)
        )
        self.pl_fields = self.fields_pl.to_xarray().rename(self.pl_names).isel(number=0)

        self.sfc_fields.coords["time"] = [
            datetime.timedelta(hours=hour) for hour in self.lagged
        ]
        self.pl_fields.coords["time"] = [
            datetime.timedelta(hours=hour) for hour in self.lagged
        ]

    def create_training_xarray(self):
        # Combine lagged and future timedeltas
        self.time_deltas = [
            datetime.timedelta(hours=h)
            for h in self.lagged
            + [
                hour
                for hour in range(
                    self.hour_steps, self.lead_time + self.hour_steps, self.hour_steps
                )
            ]
        ]
        datetimes = [self.start_date() + time_delta for time_delta in self.time_deltas]
        forcings_numpy = self.cml_variables(datetimes)
        # Create an empty training dataset that has all the variables from sfc_fields
        # and pl_fields but nans over the dimensions
        # batch, time, lat, lon, level
        # This is so we can merge the forcings dataset with the training dataset
        # and then drop the batch dimension
        empty_dataset = xr.Dataset(
            {
                "toa_incident_solar_radiation": (
                    ["batch", "time", "lat", "lon"],
                    forcings_numpy[0:1, :, :, :],
                ),
                "year_progress_sin": (["batch", "time"], forcings_numpy[1:2, :, 0, 0]),
                "year_progress_cos": (["batch", "time"], forcings_numpy[2:3, :, 0, 0]),
                "day_progress_sin": (
                    ["batch", "time", "lon"],
                    forcings_numpy[3:4, :, 0, :],
                ),
                "day_progress_cos": (
                    ["batch", "time", "lon"],
                    forcings_numpy[4:5, :, 0, :],
                ),
                "geopotential_at_surface": (
                    ["lat", "lon"],
                    np.squeeze(
                        self.sfc_fields["geopotential_at_surface"].values[0, 0, :, :]
                    ),
                ),
                "land_sea_mask": (
                    ["lat", "lon"],
                    np.squeeze(self.sfc_fields["land_sea_mask"].values[0, 0, :, :]),
                ),
            },
            coords={
                "batch": self.sfc_fields.coords["batch"],
                "time": self.time_deltas,
                "lat": self.sfc_fields.coords["lat"],
                "lon": self.sfc_fields.coords["lon"],
                "level": self.pl_fields.coords["level"],
            },
        )
        self.sfc_fields = self.sfc_fields.drop_vars(
            ["geopotential_at_surface", "land_sea_mask"]
        )

        # Create a training dataset with all the variables from sfc_fields and pl_fields
        # and the forcings dataset
        # and then drop the batch dimension
        self.training_xarray = (
            empty_dataset.combine_first(self.sfc_fields)
            .combine_first(self.pl_fields)
            .drop_vars(["batch"])
        )

    def run(self):
        with self.timer("Building model"):
            self.load_model()
        # all_fields = self.all_fields.to_xarray()

        with self.timer("Creating data"):
            self.create_graphcast_inputs()
            self.create_training_xarray()

            input_xr, template, forcings = data_utils.extract_inputs_targets_forcings(
                self.training_xarray,
                target_lead_times=self.time_deltas[len(self.lagged) :],
                **dataclasses.asdict(self.task_config),
            )

        with self.timer("Doing full rollout prediction in JAX"):
            output = self.model(
                rng=jax.random.PRNGKey(0),
                inputs=input_xr,
                targets_template=template,
                forcings=forcings,
            )

        # Write data to target location
        translate_dict = {
            "2t": "t2m",
            "10u": "u10",
            "10v": "v10",
        }

        self.task_config.target_variables

        output["total_precipitation_6hr"] = output.data_vars[
            "total_precipitation_6hr"
        ].cumsum(dim="time")

        self.all_fields = self.all_fields.order_by(
            valid_datetime="descending",
            param_level=self.ordering,
            remapping={"param_level": "{param}{levelist}"},
        )

        for t in range(self.lead_time // self.hour_steps):
            for k, fs in enumerate(
                self.all_fields[: len(self.all_fields) // len(self.lagged)]
            ):
                name = fs["shortName"]
                level = fs["level"]
                if level != 0:
                    param = self.pl_names.get(name, name)
                    if param in self.task_config.target_variables:
                        self.write(
                            output.isel(time=t)
                            .sel(level=level)
                            .data_vars[param]
                            .values,
                            template=fs,
                            step=(t + 1) * self.hour_steps,
                        )
                else:
                    sfc_name = translate_dict.get(name, name)
                    param = self.sfc_names.get(sfc_name, sfc_name)
                    if param in self.task_config.target_variables:
                        self.write(
                            output.isel(time=t).data_vars[param].values,
                            template=fs,
                            step=(t + 1) * self.hour_steps,
                        )

    def patch_retrieve_request(self, r):
        if r.get("class", "od") != "od":
            return

        if r.get("type", "an") != "an":
            return

        if r.get("stream", "oper") != "oper":
            return

        time = r.get("time", 12)

        r["stream"] = {
            0: "oper",
            6: "scda",
            12: "oper",
            18: "scda",
        }[time]


model = GraphcastModel
