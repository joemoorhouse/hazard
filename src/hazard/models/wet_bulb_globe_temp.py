"""Module for calculating indicators of days exceeding wet-bulb globe temperature (WBGT) thresholds."""

import logging
from typing_extensions import Dict

import numpy as np
import xarray as xr

from hazard.inventory import HazardResource

from hazard.models.multi_year_average import (
    ThresholdBasedAverageIndicator,
)
from hazard.models.indicator_conventions import (
    indicator_description,
    threshold_indicator_resource,
)

logger = logging.getLogger(__name__)


class DaysWBGTAboveIndicator(ThresholdBasedAverageIndicator):
    """Days per year with daily-mean and daily-maximum wet-bulb globe temperature (WBGT) above thresholds.

    Two indicators are produced, "mean" and "max", from the daily mean and daily maximum
    of near-surface temperature respectively, the water vapour partial pressure being
    taken to be constant through the day. Together these allow the hours of work above
    each threshold to be estimated for a model of hourly WBGT such as the ILO's 4+4+4
    method (see wet_bulb_globe_temp.md).
    """

    quantities = ["tas", "tasmax", "hurs"]

    # WBGT thresholds in °C: 2 °C steps where the ILO work-capacity curves are steep, 4 °C where flat
    default_thresholds = [20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44]

    def _single_year_arrays(
        self, inputs: Dict[str, xr.DataArray]
    ) -> Dict[str, xr.DataArray]:
        tas_c = inputs["tas"] - 273.15  # convert from K to °C
        # water vapour partial pressure in hPa, from the daily mean temperature and relative
        # humidity; taken to be constant through the day, so used for the maximum as well
        hurs = inputs["hurs"].clip(0.0, 100.0)
        vpp = (hurs / 100.0) * 6.105 * np.exp((17.27 * tas_c) / (237.7 + tas_c))
        wbgt_mean = 0.567 * tas_c + 0.393 * vpp + 3.94
        mean = self._days_above(wbgt_mean)
        del tas_c, hurs, wbgt_mean  # limit peak memory
        wbgt_max = 0.567 * (inputs["tasmax"] - 273.15) + 0.393 * vpp + 3.94
        return {"mean": mean, "max": self._days_above(wbgt_max)}

    def _resources(self) -> Dict[str, HazardResource]:
        description = indicator_description("wet_bulb_globe_temp.md", self.run)
        return {
            key: threshold_indicator_resource(
                run=self.run,
                thresholds=self.thresholds,
                indicator_id=f"days_wbgt_{key}_above",
                path_name=f"days_wbgt_{key}_above",
                display_name=f"Days with daily-{'mean' if key == 'mean' else 'maximum'} "
                + "wet-bulb globe temperature above threshold in °C ({gcm})",
                display_group="Days with wet-bulb globe temperature above threshold in °C",
                description=description,
                units="days/year",
                max_value=365.0,
            )
            for key in ["mean", "max"]
        }
