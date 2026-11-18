[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/belski21-growl/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/belski21-growl/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# gta6-twitch-ccu-forecast-01

Forecast GTA VI's launch-day Twitch viewers hour by hour (hours 2–24),
as five quantiles per hour at levels [0.05, 0.25, 0.5, 0.75, 0.95].

## Model (round 2)

Comparable-launch curve, anchored on the case's own first hour, with
scale-aware members chosen by leave-one-out (LOO) validation on the 11
rolling-midnight comparables + the 4 practice launches:

1. **Filter comparables** to rolling-midnight-local console launches
   (unlock text says "midnight local"; exclude early-access-contaminated
   and weak-tier launches, and the target game's own pre-release category
   history). Simultaneous-global launches have a totally different day
   shape (tiny hour-1 print, then a 10–100x explosion) and are excluded.
2. **Level anchor**: geometric mean of the hour-1 print and the hour
   before release day (23:00 UTC the day before, when Europe has already
   unlocked). Hour 1 alone is a noisy single-hour print.
3. **Per hour** h = 2..24, two members estimate the median
   log(viewers_h / anchor):
   - Gaussian-kernel **weighted median** of the comparables' log-ratios
     (bandwidth 0.4 on |log anchor ratio| — same-size launches weigh
     most);
   - **OLS log-log** regression log(viewers_h) ~ a + b·log(anchor),
     evaluated at the case anchor (slope clamped [-0.5, 1.5]) — the only
     member that extrapolates to GTA VI's scale, above every comparable
     (peak growth shrinks with scale: 32k-anchor launches peak ~3x,
     851k-anchor ELDEN RING peaks 0.98x).
   Forecast median = mean of the two members.
4. **Pre-checkpoint slope adjustment**: the case's own log-slope over the
   3 hours before day start, added at half weight to hours 2–12.
5. **Spread** sigma_h = IQR/1.349 of the log-ratios, floored at 0.35 and
   capped at 0.70.
6. Forecast: `base * exp(median_h + z_tau * sigma_h)`, quantiles capped
   at 1e9 viewers. Fallback to the house persistence baseline if fewer
   than 5 rolling comparables exist.

Rejected by LOO (tested, kept out): global per-hour bias correction,
local-linear member, spread from LOO residuals, channels as a second
kernel dimension, pre-day slope as a similarity weight, evening slope
damping. No practice-answer replay — the model never sees a practice
launch's answer.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, deterministic, no network.

## Practice validation (local, answers from SullyGnome)

| case | loss | baseline loss |
| ---- | ---- | ------------- |
| p00 Marvel's Wolverine | 0.174 | 0.744 |
| p01 Red Dead Redemption 2 | 0.109 | 0.287 |
| p02 God of War Ragnarök | 0.120 | 0.215 |
| p03 Marvel's Spider-Man 2 | 0.230 | 0.303 |
| **mean** | **0.158** | **0.387** |

Round 1 scored 0.223 mean; round 2 improves it by 29% (LOO mean 0.2175
vs 0.2948 round 1). Coverage-90 mean 0.96, typical miss ~27% (round 1:
0.86 / 42%). Every case beats the house starter baseline.

Outputs are byte-identical across repeated runs; runtime well under
8 s per case. See `build-log.md` for the full validation table.
