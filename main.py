# SPDX-License-Identifier: MIT
"""Forecast GTA VI launch-day Twitch viewers, hour by hour (quantiles).

Model: comparable-launch curve, anchored on the case's own first hour.
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
  - For each forecast hour h = 2..24, take the median across comparables
    of log(viewers_h / anchor), and a log spread sigma_h = IQR/1.349 of
    the same log-ratios, floored at 0.35 and capped at 0.70 (the
    cross-launch dispersion is large; a narrower spread is overconfident).
  - The case forecast is base * exp(median_h + z_tau * sigma_h), where
    base is the case's own geometric-mean anchor and z_tau the standard
    normal quantile at each scored level.
  - Fallbacks, in order: rolling-midnight comparables; all comparables
    that are not early-access-contaminated / weak; the house persistence
    baseline (every hour at the hour-1 print, spread 0.15*sqrt(h-1)).

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


def comparable_log_ratios(case, comps):
    """Per-hour lists of log(viewers_h / anchor) over the comparables."""
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
        for h in range(2, 25):
            value = hourly.get(day_start + timedelta(hours=h - 1))
            if value is not None and value > 0:
                per_hour[h].append(math.log(value / anchor))
    return per_hour, used


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
    if day_start is not None:
        base = level_anchor(history_hourly, day_start)
    if base is None:
        base = p1

    comps = case.get("comparables") or []
    rolling = [c for c in comps if is_rolling_launch(case, c)]
    per_hour, used = comparable_log_ratios(case, rolling)
    if used < MIN_COMPARABLES:
        usable = [c for c in comps if is_usable_launch(case, c)]
        per_hour, used = comparable_log_ratios(case, usable)
    if used < MIN_COMPARABLES:
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                          for h in range(2, 25)]}

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        ratios = sorted(per_hour.get(h, []))
        if not ratios:
            hours.append({"hour": h, "q": baseline_quantiles(p1, h)})
            continue
        med = median(ratios)
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
