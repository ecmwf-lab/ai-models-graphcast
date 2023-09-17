# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

GRIB_TO_XARRAY_SFC = {
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tp": "total_precipitation_6hr",
    "z": "geopotential_at_surface",
    "lsm": "land_sea_mask",
    "latitude": "lat",
    "longitude": "lon",
    # "step": "batch",
    "valid_time": "datetime",
}

GRIB_TO_XARRAY_PL = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
    "isobaricInhPa": "level",
    "latitude": "lat",
    "longitude": "lon",
    # "step": "batch",
    "valid_time": "datetime",
}


GRIB_TO_CF = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
}
