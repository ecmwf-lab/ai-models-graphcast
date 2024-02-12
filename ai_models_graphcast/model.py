# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import dataclasses
import datetime
import functools
import gc
import logging
import os
from functools import cached_property

import xarray
from ai_models.model import Model

from .input import create_training_xarray
from .output import save_output_xarray

LOG = logging.getLogger(__name__)


try:
    import haiku as hk
    import jax
    from graphcast import (
        autoregressive,
        casting,
        checkpoint,
        data_utils,
        graphcast,
        normalization,
    )
except ModuleNotFoundError as e:
    msg = "You need to install Graphcast from git to use this model. See README.md for details."
    LOG.error(msg)
    raise ModuleNotFoundError(f"{msg}\n{e}")


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

    forcing_variables = [
        "toa_incident_solar_radiation",
        # Not calling julian day and day here, due to differing assumptions with Deepmind
        # Those forcings are created by graphcast.data_utils
    ]

    use_an = False

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
            fn,
            model_config=self.model_config,
            task_config=self.task_config,
        )

    # Always pass params and state, so the usage below are simpler
    def _with_params(self, fn):
        return functools.partial(fn, params=self.params, state=self.state)

    # Deepmind models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by the rollout code, and generally simpler.
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
                    predictor,
                    gradient_checkpointing=True,
                )
                return predictor

            @hk.transform_with_state
            def run_forward(
                model_config,
                task_config,
                inputs,
                targets_template,
                forcings,
            ):
                predictor = construct_wrapped_graphcast(model_config, task_config)
                return predictor(
                    inputs,
                    targets_template=targets_template,
                    forcings=forcings,
                )

            with open(get_path(self.download_files[0]), "rb") as f:
                self.ckpt = checkpoint.load(f, graphcast.CheckPoint)
                self.params = self.ckpt.params
                self.state = {}

                self.model_config = self.ckpt.model_config
                self.task_config = self.ckpt.task_config

                LOG.info("Model description: %s", self.ckpt.description)
                LOG.info("Model license: %s", self.ckpt.license)

            jax.jit(self._with_configs(run_forward.init))
            self.model = self._drop_state(
                self._with_params(jax.jit(self._with_configs(run_forward.apply)))
            )

    @cached_property
    def start_date(self) -> "datetime":
        return self.all_fields.order_by(valid_datetime="descending")[0].datetime

    def run(self):
        # We ignore 'tp' so that we make sure that step 0 is a field of zero values
        self.write_input_fields(self.fields_sfc, ignore=["tp"], accumulations=["tp"])
        self.write_input_fields(self.fields_pl)

        with self.timer("Building model"):
            self.load_model()

        with self.timer("Creating input data (total)"):
            with self.timer("Creating training data"):
                training_xarray, time_deltas = create_training_xarray(
                    fields_sfc=self.fields_sfc,
                    fields_pl=self.fields_pl,
                    lagged=self.lagged,
                    start_date=self.start_date,
                    hour_steps=self.hour_steps,
                    lead_time=self.lead_time,
                    forcing_variables=self.forcing_variables,
                    constants=self.override_constants,
                    timer=self.timer,
                )

            gc.collect()

            if self.debug:
                training_xarray.to_netcdf("training_xarray.nc")

            with self.timer("Extracting input targets"):
                (
                    input_xr,
                    template,
                    forcings,
                ) = data_utils.extract_inputs_targets_forcings(
                    training_xarray,
                    target_lead_times=[
                        f"{int(delta.days * 24 + delta.seconds/3600):d}h"
                        for delta in time_deltas[len(self.lagged) :]
                    ],
                    **dataclasses.asdict(self.task_config),
                )

            if self.debug:
                input_xr.to_netcdf("input_xr.nc")
                forcings.to_netcdf("forcings_xr.nc")

        with self.timer("Doing full rollout prediction in JAX"):
            output = self.model(
                rng=jax.random.PRNGKey(0),
                inputs=input_xr,
                targets_template=template,
                forcings=forcings,
            )

            if self.debug:
                output.to_netcdf("output.nc")

        with self.timer("Saving output data"):
            save_output_xarray(
                output=output,
                write=self.write,
                target_variables=self.task_config.target_variables,
                all_fields=self.all_fields,
                ordering=self.ordering,
                lead_time=self.lead_time,
                hour_steps=self.hour_steps,
                lagged=self.lagged,
            )

    def patch_retrieve_request(self, r):
        if r.get("class", "od") != "od":
            return

        if r.get("type", "an") not in ("an", "fc"):
            return

        if r.get("stream", "oper") not in ("oper", "scda"):
            return

        if self.use_an:
            r["type"] = "an"
        else:
            r["type"] = "fc"

        time = r.get("time", 12)

        r["stream"] = {
            0: "oper",
            6: "scda",
            12: "oper",
            18: "scda",
        }[time]

    def parse_model_args(self, args):
        import argparse

        parser = argparse.ArgumentParser("ai-models graphcast")
        parser.add_argument("--use-an", action="store_true")
        parser.add_argument("--override-constants")
        return parser.parse_args(args)


model = GraphcastModel
