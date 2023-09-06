# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging

import climetlab as cml
import numpy as np
import xarray as xr

from .convert import GRIB_TO_XARRAY_PL, GRIB_TO_XARRAY_SFC

LOG = logging.getLogger(__name__)


def forcing_variables_numpy(sample, forcing_variables, dates):
    """Generate variables from climetlabs

    Args:
        date (datetime): Datetime of current time step in forecast
        params (List[str]): Parameters to calculate as constants

    Returns:
        torch.Tensor: Tensor with constants
    """
    ds = cml.load_source(
        "constants",
        sample,
        date=dates,
        param=forcing_variables,
    )

    return (
        ds.order_by(param=forcing_variables, valid_datetime="ascending")
        .to_numpy()
        .reshape(len(forcing_variables), len(dates), 721, 1440)
    )


def create_training_xarray(
    *,
    fields_sfc,
    fields_pl,
    lagged,
    start_date,
    hour_steps,
    lead_time,
    forcing_variables,
    constants,
):
    LOG.info("Creating training dataset")
    # Create Input dataset

    sfc_fields_xarray = (
        fields_sfc.to_xarray().rename(GRIB_TO_XARRAY_SFC).isel(number=0, surface=0)
    )

    pl_fields_xarray = fields_pl.to_xarray().rename(GRIB_TO_XARRAY_PL).isel(number=0)

    sfc_fields_xarray.coords["time"] = [
        datetime.timedelta(hours=hour) for hour in lagged
    ]

    pl_fields_xarray.coords["time"] = [
        datetime.timedelta(hours=hour) for hour in lagged
    ]

    # Combine lagged and future timedeltas
    time_deltas = [
        datetime.timedelta(hours=h)
        for h in lagged
        + [hour for hour in range(hour_steps, lead_time + hour_steps, hour_steps)]
    ]
    datetimes = [start_date() + time_delta for time_delta in time_deltas]
    forcing_numpy = forcing_variables_numpy(fields_sfc, forcing_variables, datetimes)
    # Create an empty training dataset that has all the variables from sfc_fields
    # and pl_fields but nans over the dimensions
    # batch, time, lat, lon, level
    # This is so we can merge the forcing dataset with the training dataset
    # and then drop the batch dimension
    empty_dataset = xr.Dataset(
        {
            "toa_incident_solar_radiation": (
                ["batch", "time", "lat", "lon"],
                forcing_numpy[0:1, :, :, :],
            ),
            "geopotential_at_surface": (
                ["lat", "lon"],
                np.squeeze(
                    sfc_fields_xarray["geopotential_at_surface"].values[0, 0, :, :]
                ),
            ),
            "land_sea_mask": (
                ["lat", "lon"],
                np.squeeze(sfc_fields_xarray["land_sea_mask"].values[0, 0, :, :]),
            ),
        },
        coords={
            "batch": sfc_fields_xarray.coords["batch"],
            "time": time_deltas,
            "lat": sfc_fields_xarray.coords["lat"],
            "lon": sfc_fields_xarray.coords["lon"],
            "level": pl_fields_xarray.coords["level"],
        },
    )
    sfc_fields_xarray = sfc_fields_xarray.drop_vars(
        [
            "geopotential_at_surface",
            "land_sea_mask",
        ]
    )

    # Create a training dataset with all the variables from sfc_fields and pl_fields
    # and the forcing dataset
    # and then drop the batch dimension
    training_xarray = (
        empty_dataset.combine_first(sfc_fields_xarray)
        .combine_first(pl_fields_xarray)
        .drop_vars(["batch"])
    )

    # Ensure we have the right order
    training_xarray = training_xarray.transpose(
        "batch",
        "time",
        "level",
        "lat",
        "lon",
    )

    # cfgrib sorts the pressure levels in descending order
    # we want them in ascending order

    training_xarray = training_xarray.reindex(
        level=sorted(training_xarray.level.values)
    )

    # And we want the grid south to north
    training_xarray = training_xarray.reindex(lat=sorted(training_xarray.lat.values))

    # # Shift times to timedelta 0
    training_xarray["time"] = [
        time_delta - time_deltas[0] for time_delta in time_deltas
    ]

    # Add datetimes as coordinates
    training_xarray.coords["datetime"].data[0, :] = datetimes

    if constants:
        # Add geopotential_at_surface and land_sea_mask back in
        x = xr.load_dataset(constants)

        for patch in ("geopotential_at_surface", "land_sea_mask"):
            LOG.info("PATCHING %s", patch)
            training_xarray[patch] = x[patch]

    return training_xarray, time_deltas
