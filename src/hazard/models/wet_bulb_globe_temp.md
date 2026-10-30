Days per year for which the daily-mean or daily-maximum 'Wet-Bulb Globe Temperature' (WBGT) is above a threshold specified in °C:

$$
I^\text{mean}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^\text{WBGT,mean}_d > T^\text{ref}}, \qquad I^\text{max}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^\text{WBGT,max}_d > T^\text{ref}}
$$

$d$ is the day index, $n_y$ is the number of days in the year and $T^\text{ref}$ is the threshold (reference) temperature. The indicators are the expectations of $I^\text{mean}$ and $I^\text{max}$, estimated by averaging over a 20-year window. For each indicator, the index dimension of the array is the threshold.

The (in-shade) WBGT is calculated from the daily mean and daily maximum near-surface temperature in °C, $T^\text{avg}_d$ and $T^\text{max}_d$, and the water vapour partial pressure in hPa, $p^\text{vapour}_d$:

$$
T^\text{WBGT,mean}_d = 0.567 \times T^\text{avg}_d + 0.393 \times p^\text{vapour}_d + 3.94, \qquad T^\text{WBGT,max}_d = 0.567 \times T^\text{max}_d + 0.393 \times p^\text{vapour}_d + 3.94
$$

The water vapour partial pressure is calculated from the daily mean temperature and relative humidity $h^\text{relative}_d$ (in %) and is taken to be constant through the day, so that the same value is used for the daily maximum:

$$
p^\text{vapour}_d = \frac{h^\text{relative}_d}{100} \times 6.105 \times \exp \left( \frac{17.27 \times T^\text{avg}_d}{237.7 + T^\text{avg}_d} \right)
$$

The daily-mean and daily-maximum WBGT can be used to model hourly WBGT during the working day, for example with the '4+4+4' method of the International Labour Organization (ILO, 'Working on a warmer planet', 2019, Appendix I): of 12 daylight hours, 4 are close to the maximum WBGT, 4 close to the mean WBGT and 4 halfway between the two. For a model of $m$ hours at the maximum and $n$ hours at the mean, the hours per year with WBGT above a threshold are given exactly by $m \, i^\text{max} + n \, i^\text{mean}$. For the 4+4+4 method these are bounded by $4 \, i^\text{max} + 8 \, i^\text{mean}$ and $8 \, i^\text{max} + 4 \, i^\text{mean}$, and approximated by $6 \, i^\text{max} + 6 \, i^\text{mean}$. Differences between successive thresholds give the hours within WBGT bands, to which exposure-response functions for work capacity (such as those of the ILO for work intensities of 200 W, 300 W and 400 W) can be applied.
