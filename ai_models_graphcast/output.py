# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from .convert import GRIB_TO_CF, GRIB_TO_XARRAY_PL, GRIB_TO_XARRAY_SFC

LOG = logging.getLogger(__name__)


def save_output_xarray(
    *,
    output,
    target_variables,
    write,
    all_fields,
    ordering,
    lead_time,
    hour_steps,
    lagged,
):
    LOG.info("Converting output xarray to GRIB and saving")

    output["total_precipitation_6hr"] = output.data_vars[
        "total_precipitation_6hr"
    ].cumsum(dim="time")

    all_fields = all_fields.order_by(
        valid_datetime="descending",
        param_level=ordering,
        remapping={"param_level": "{param}{levelist}"},
    )

    for time in range(lead_time // hour_steps):
        for fs in all_fields[: len(all_fields) // len(lagged)]:
            name, level = fs["shortName"], fs["level"]

            if level != 0:
                param = GRIB_TO_XARRAY_PL.get(name, name)
                if param in target_variables:
                    write(
                        output.isel(time=time).sel(level=level).data_vars[param].values,
                        template=fs,
                        step=(time + 1) * hour_steps,
                    )
                else:
                    LOG.warn(f"Skipping {name} as not in target variables")
            else:
                sfc_name = GRIB_TO_CF.get(name, name)
                param = GRIB_TO_XARRAY_SFC.get(sfc_name, sfc_name)
                if param in target_variables:
                    write(
                        output.isel(time=time).data_vars[param].values,
                        template=fs,
                        step=(time + 1) * hour_steps,
                    )
                else:
                    LOG.warn(f"Skipping {name} as not in target variables")
