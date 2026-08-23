"""Maps a (valence, arousal) pair to a music descriptor by interpolating
between the two nearest named regions in config/va_regions.yaml."""

import math
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "va_regions.yaml"

_regions_cache: list[dict[str, Any]] | None = None


def load_regions() -> list[dict[str, Any]]:
    global _regions_cache
    if _regions_cache is None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _regions_cache = data["regions"]
    return _regions_cache


def _distance(v1: float, a1: float, v2: float, a2: float) -> float:
    return math.hypot(v1 - v2, a1 - a2)


def nearest_regions(valence: float, arousal: float, n: int = 2) -> list[tuple[dict, float]]:
    regions = load_regions()
    scored = [
        (r, _distance(valence, arousal, r["valence"], r["arousal"]))
        for r in regions
    ]
    scored.sort(key=lambda pair: pair[1])
    return scored[:n]


def map_va_to_descriptor(valence: float, arousal: float) -> dict[str, Any]:
    """Returns a descriptor dict blended from the two nearest regions,
    weighted by inverse distance. Falls back to the single nearest region
    if the point sits exactly on it."""
    top2 = nearest_regions(valence, arousal, n=2)
    (r1, d1), (r2, d2) = top2[0], top2[1]

    if d1 < 1e-6:
        primary, secondary, w1 = r1, r1, 1.0
    else:
        w1 = (1 / d1) / (1 / d1 + 1 / max(d2, 1e-6))
        primary, secondary = r1, r2

    return {
        "primary_region": primary["name"],
        "secondary_region": secondary["name"],
        "blend_weight": round(w1, 2),
        "tempo": primary["descriptor"]["tempo"],
        "instruments": list(dict.fromkeys(
            primary["descriptor"]["instruments"] + secondary["descriptor"]["instruments"]
        )),
        "dynamics": list(dict.fromkeys(
            primary["descriptor"]["dynamics"] + secondary["descriptor"]["dynamics"]
        )),
        "mood_words": list(dict.fromkeys(
            primary["descriptor"]["mood_words"] + secondary["descriptor"]["mood_words"]
        )),
    }
