# SPDX-License-Identifier: MIT
"""Forecast GTA VI launch-day Twitch viewers, hour by hour (quantiles).

Round 2 model: comparable-launch curve, anchored on the case's own first
hour, with scale-aware members chosen by leave-one-out (LOO) validation
on the 11 rolling-midnight comparables + the 4 practice launches
(15 judge cases; practice answers are public SullyGnome history, used
only for offline validation, never at runtime).

  - Filter comparables to rolling-midnight-local console launches: the
    unlock text must say "midnight local", and the launch must not be
    early-access-contaminated or weak-tier. Simultaneous-global launches
    (tiny hour-1 print, then a 10-100x explosion) and early-access launches
    have a completely different day shape and are excluded. The target
    game's own pre-release category history ("Not a launch") is excluded.
  - Each comparable is normalised by its own level anchor: the geometric
    mean of its hour-1 print and the hour before release day (23:00 UTC
    the day before, when Europe has already unlocked). Hour 1 alone is a
    noisy single-hour print; the hour before already reflects the
    Europe unlock, so the geometric mean is a stabler level.
  - For each forecast hour h = 2..24, two members estimate the median
    log(viewers_h / anchor):
      (1) a Gaussian-kernel weighted median of the comparables' log-ratios,
          weighted by scale similarity: w = exp(-0.5 * (dx / 0.4) ** 2)
          with dx = log(anchor_comp / anchor_case). A launch ten times
          smaller than the case carries ~1/25 the weight of a same-size
          one. (Unweighted median: LOO 0.2948, practice 0.2233.)
      (2) an OLS log-log regression across comparables,
          log(viewers_h) ~ a + b * log(anchor), evaluated at the case's
          anchor (slope clamped to [-0.5, 1.5]). Peak growth shrinks with
          scale in the data (32k-anchor launches peak ~3x, 851k-anchor
          ELDEN RING peaks 0.98x), so the OLS member is the only one that
          extrapolates to GTA VI's scale, far above every comparable.
    The forecast median is the mean of the two members.
  - Pre-checkpoint slope adjustment: the case's own log-slope over the
    3 hours before day start (from history_hourly), added at half weight
    to hours 2..12. A steeply falling pre-day curve (Europe already
    unlocked and going to bed) predicts lower early hours.
  - Spread sigma_h = IQR/1.349 of the (unweighted) log-ratios, floored at
    0.35 and capped at 0.70.
  - The case forecast is base * exp(median_h + z_tau * sigma_h), where
    base is the case's own geometric-mean anchor and z_tau the standard
    normal quantile at each scored level.
  - Fallbacks, in order: rolling-midnight comparables; all comparables
    that are not early-access-contaminated / weak; the house persistence
    baseline (every hour at the hour-1 print, spread 0.15*sqrt(h-1)).

What LOO rejected (tested, kept out): a global per-hour bias correction
from LOO residuals (LOO 0.218 -> 0.235, practice unchanged); a
local-linear member (LOO worse); spread from LOO residuals (LOO worse,
practice only marginally better); channels as a second kernel dimension
(LOO 0.220 -> 0.229..0.270); pre-day slope as a similarity weight
(no gain); evening slope damping (worse). No practice-answer replay:
the model never sees a practice launch's answer.

Offline validation (LOO judge + practice, 18 Nov 2026):
  LOO mean 0.2175 (r1: 0.2948); practice mean 0.1582 (r1: 0.2233):
  p00 0.1743, p01 0.1089, p02 0.1197, p03 0.2299 — every case beats the
  house starter baseline (0.7437 / 0.2868 / 0.2146 / 0.3033).
  Coverage-90 mean 0.96, typical miss ~27% (r1: 0.86 / 42%).

Python 3.13, standard library only, deterministic, no network, no
randomness. Reads every *.in in $MAKEFUN_INPUT_DIR (else /input) and
writes one *.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA_FLOOR = 0.35
SIGMA_CAP = 0.70
BASELINE_SIGMA = 0.15
MIN_COMPARABLES = 5
KERNEL_BW = 0.4
OLS_SLOPE_MIN = -0.5
OLS_SLOPE_MAX = 1.5
SLOPE_K = 0.5
SLOPE_HMAX = 12


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up (same as the starter's round2)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def quantile(sorted_values, q):
    """Linear-interpolation quantile of a sorted list (deterministic)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def median(values):
    return quantile(sorted(values), 0.5)


def weighted_median(pairs):
    """Median of [(value, weight)]; ties break toward the lower value."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2.0:
            return value
    return ordered[-1][0]


def is_rolling_launch(case, comp):
    """True for a rolling-midnight-local console launch comparable."""
    unlock = (comp.get("unlock") or "").lower()
    if "midnight local" not in unlock:
        return False
    if "early-access-contaminated" in unlock or "tier: weak" in unlock:
        return False
    if "not a launch" in unlock:
        return False
    target = (case.get("game") or {}).get("name", "").lower()
    if target and target in (comp.get("name") or "").lower():
        return False
    return True


def is_usable_launch(case, comp):
    """Looser filter used only if too few rolling launches exist."""
    unlock = (comp.get("unlock") or "").lower()
    if "early-access-contaminated" in unlock or "tier: weak" in unlock:
        return False
    if "not a launch" in unlock:
        return False
    target = (case.get("game") or {}).get("name", "").lower()
    if target and target in (comp.get("name") or "").lower():
        return False
    return True


def hourly_map(comp):
    """Map hour-start datetime -> viewers for one comparable."""
    out = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        try:
            out[parse_time(row["from"])] = float(viewers)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def level_anchor(hourly, day_start):
    """Geometric mean of the hour-1 print and the hour before day start.

    Returns None when the hour-1 print itself is missing.
    """
    p1 = hourly.get(day_start)
    if p1 is None or p1 <= 0:
        return None
    prev = hourly.get(day_start - timedelta(hours=1))
    if prev is None or prev <= 0:
        return float(p1)
    return math.sqrt(p1 * prev)


def pre_day_slope(history_hourly, day_start):
    """Log-slope per hour over the 3 hours before day start.

    log(v(-1h) / v(-3h)) / 2 from the case's own history_hourly; 0.0 when
    either point is missing. Negative when the pre-day curve is falling
    (Europe unlocked, going to bed) into the UTC day.
    """
    v1 = history_hourly.get(day_start - timedelta(hours=1))
    v3 = history_hourly.get(day_start - timedelta(hours=3))
    if v1 is None or v3 is None or v1 <= 0 or v3 <= 0:
        return 0.0
    return math.log(v1 / v3) / 2.0


def comparable_data(case, comps):
    """Per-hour data over the comparables.

    Returns (per_hour, used) where per_hour[h] holds one tuple per
    comparable with an hour-h print: (log-ratio y = log(viewers_h /
    anchor), log anchor, log viewers_h). The scale distance
    x = log(anchor_comp / anchor_case) is log anchor minus the case's
    log anchor, computed by the caller.
    """
    per_hour = {h: [] for h in range(2, 25)}
    used = 0
    for comp in comps:
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hourly = hourly_map(comp)
        anchor = level_anchor(hourly, day_start)
        if anchor is None:
            continue
        used += 1
        log_anchor = math.log(anchor)
        for h in range(2, 25):
            value = hourly.get(day_start + timedelta(hours=h - 1))
            if value is not None and value > 0:
                per_hour[h].append((math.log(value / anchor), log_anchor,
                                    math.log(value)))
    return per_hour, used


def ols_log_ratio(rows, log_target_anchor):
    """OLS log-log member: log(viewers_h) ~ a + b*log(anchor).

    Returns the predicted log(viewers_h / anchor_case), or None when the
    fit is degenerate. Slope clamped to [-0.5, 1.5].
    """
    n = len(rows)
    if n < 3:
        return None
    mean_x = sum(r[1] for r in rows) / n
    mean_y = sum(r[2] for r in rows) / n
    sxx = sum((r[1] - mean_x) ** 2 for r in rows)
    if sxx <= 0:
        return None
    sxy = sum((r[1] - mean_x) * (r[2] - mean_y) for r in rows)
    slope = sxy / sxx
    slope = max(OLS_SLOPE_MIN, min(OLS_SLOPE_MAX, slope))
    intercept = mean_y - slope * mean_x
    return intercept + slope * log_target_anchor - log_target_anchor


def baseline_quantiles(p1, h):
    """House persistence baseline (starter): flat median, widening spread."""
    return [round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # The case's own level anchor: hour-1 print + hour before day start.
    history_hourly = {}
    for row in case.get("history_hourly") or []:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        try:
            history_hourly[parse_time(row["from"])] = float(viewers)
        except (KeyError, TypeError, ValueError):
            continue
    try:
        day_start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        day_start = None
    base = None
    slope = 0.0
    if day_start is not None:
        base = level_anchor(history_hourly, day_start)
        slope = pre_day_slope(history_hourly, day_start)
    if base is None:
        base = p1

    comps = case.get("comparables") or []
    rolling = [c for c in comps if is_rolling_launch(case, c)]
    per_hour, used = comparable_data(case, rolling)
    if used < MIN_COMPARABLES:
        usable = [c for c in comps if is_usable_launch(case, c)]
        per_hour, used = comparable_data(case, usable)
    if used < MIN_COMPARABLES:
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                          for h in range(2, 25)]}

    log_base = math.log(base)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        rows = per_hour.get(h, [])
        if not rows:
            hours.append({"hour": h, "q": baseline_quantiles(p1, h)})
            continue
        # Member 1: Gaussian-kernel weighted median of log-ratios.
        pairs = []
        for log_ratio, log_anchor, _ in rows:
            dx = log_anchor - log_base
            pairs.append((log_ratio, math.exp(-0.5 * (dx / KERNEL_BW) ** 2)))
        members = [weighted_median(pairs)]
        # Member 2: OLS log-log, evaluated at the case anchor.
        ols = ols_log_ratio(rows, log_base)
        if ols is not None:
            members.append(ols)
        med = sum(members) / len(members)
        # Pre-checkpoint slope adjustment, early hours only.
        if h <= SLOPE_HMAX:
            med += SLOPE_K * slope
        ratios = sorted(r[0] for r in rows)
        iqr = quantile(ratios, 0.75) - quantile(ratios, 0.25)
        sigma = iqr / 1.349 if iqr > 0 else SIGMA_FLOOR
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        qs = [round2(base * math.exp(med + z * sigma)) for z in Z]
        # Spec cap: every quantile must be > 0 and at most 1e9 viewers.
        qs = [min(max(1.0, q), 1000000000.0) for q in qs]
        qs = sorted(qs)
        hours.append({"hour": h, "q": qs})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
