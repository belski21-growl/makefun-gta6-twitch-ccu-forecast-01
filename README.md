# gta6-twitch-ccu-forecast-01

Forecast GTA VI's launch-day Twitch viewers hour by hour (hours 2–24),
as five quantiles per hour at levels [0.05, 0.25, 0.5, 0.75, 0.95].

## Model

Comparable-launch curve, anchored on the case's own first hour:

1. **Filter comparables** to rolling-midnight-local console launches
   (unlock text says "midnight local"; exclude early-access-contaminated
   and weak-tier launches, and the target game's own pre-release category
   history). Simultaneous-global launches have a totally different day
   shape (tiny hour-1 print, then a 10–100x explosion) and are excluded.
2. **Level anchor**: geometric mean of the hour-1 print and the hour
   before release day (23:00 UTC the day before, when Europe has already
   unlocked). Hour 1 alone is a noisy single-hour print.
3. **Per hour** h = 2..24: median across comparables of
   log(viewers_h / anchor); spread sigma_h = IQR/1.349 of the same
   log-ratios, floored at 0.35 and capped at 0.70.
4. Forecast: `base * exp(median_h + z_tau * sigma_h)`, quantiles capped
   at 1e9 viewers. Fallback to the house persistence baseline if fewer
   than 5 rolling comparables exist.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, deterministic, no network.

## Practice validation (local, answers from SullyGnome)

| case | loss | baseline loss |
| ---- | ---- | ------------- |
| p00 Marvel's Wolverine | 0.317 | 0.744 |
| p01 Red Dead Redemption 2 | 0.117 | 0.287 |
| p02 God of War Ragnarök | 0.228 | 0.215 |
| p03 Marvel's Spider-Man 2 | 0.231 | 0.303 |
| **mean** | **0.223** | **0.387** |

Outputs are byte-identical across repeated runs; runtime well under
8 s per case. See `build-log.md` for the full validation table.
