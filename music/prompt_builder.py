"""Turns a V/A descriptor dict into a natural-language MusicGen text prompt,
with light randomization so repeated calls into the same region don't
produce an identical prompt string every time."""

import random

_OPENERS = [
    "An instrumental piece that feels {moods}.",
    "A {moods} instrumental track.",
    "Instrumental music with a {moods} feeling.",
    "A short instrumental piece, {moods} in mood.",
]

_INSTRUMENT_TEMPLATES = [
    "Featuring {instruments}.",
    "Built around {instruments}.",
    "Led by {instruments}.",
]

_DYNAMICS_TEMPLATES = [
    "The dynamics are {dynamics}.",
    "Performed in a {dynamics} way.",
    "{dynamics_cap} dynamics throughout.",
]


def _sample(items: list[str], k: int) -> list[str]:
    k = min(k, len(items))
    return random.sample(items, k)


def build_prompt(descriptor: dict) -> str:
    moods = _sample(descriptor["mood_words"], 3)
    instruments = _sample(descriptor["instruments"], 3)
    dynamics = _sample(descriptor["dynamics"], 2)
    tempo = descriptor["tempo"]

    moods_str = ", ".join(moods)
    instruments_str = ", ".join(instruments)
    dynamics_str = ", ".join(dynamics)

    opener = random.choice(_OPENERS).format(moods=moods_str)
    instr_line = random.choice(_INSTRUMENT_TEMPLATES).format(instruments=instruments_str)
    dyn_line = random.choice(_DYNAMICS_TEMPLATES).format(
        dynamics=dynamics_str, dynamics_cap=dynamics_str.capitalize()
    )
    tempo_line = f"Tempo around {tempo}."

    return " ".join([opener, instr_line, dyn_line, tempo_line])
