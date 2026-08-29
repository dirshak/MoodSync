"""Closed-loop emotion-consistency evaluation for MoodSync's zero-shot pipeline.

For each of the ten named V/A regions we run the *deployed* pipeline
(map_va_to_descriptor -> build_prompt -> MusicGen) to render clips, then ask an
independent audio-text model (CLAP) which mood each clip actually sounds like.

The CLAP probes are deliberately worded differently from the generation prompts
(no shared template, no shared instrument/dynamics vocabulary), so a correct
match reflects audible mood rather than echoed prompt wording.

Outputs eval/results.json.
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from music.va_mapper import load_regions, map_va_to_descriptor
from music.prompt_builder import build_prompt

CLAP_ID = "laion/clap-htsat-unfused"
CLAP_SR = 48000

# Independent probes: plain affect words, no instruments/dynamics/tempo terms,
# so CLAP cannot succeed by matching the generation prompt's surface form.
PROBES = {
    "relaxed":   "relaxed, laid-back music",
    "calm":      "calm, tranquil music",
    "happy":     "happy, cheerful music",
    "excited":   "excited, thrilling music",
    "hopeful":   "hopeful, optimistic music",
    "sad":       "sad, sorrowful music",
    "bored":     "dull, monotonous music",
    "angry":     "angry, aggressive music",
    "anxious":   "anxious, uneasy music",
    "exhausted": "weary, drained music",
}


def load_clap():
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(CLAP_ID).eval()
    proc = ClapProcessor.from_pretrained(CLAP_ID)
    return model, proc


def embed_text(model, proc, texts):
    inp = proc(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        e = model.get_text_features(**inp)
    return torch.nn.functional.normalize(e, dim=-1)


def embed_audio(model, proc, wav_path):
    sr, data = wavfile.read(wav_path)
    audio = data.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != CLAP_SR:                       # CLAP expects 48 kHz
        from math import gcd
        g = gcd(int(sr), CLAP_SR)
        audio = resample_poly(audio, CLAP_SR // g, int(sr) // g)
    inp = proc(audios=[audio], sampling_rate=CLAP_SR, return_tensors="pt")
    with torch.no_grad():
        e = model.get_audio_features(**inp)
    return torch.nn.functional.normalize(e, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-per-region", type=int, default=3)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--outdir", default=str(BASE / "eval" / "clips"))
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = load_regions()
    names = [r["name"] for r in regions]
    assert set(names) == set(PROBES), "region/probe mismatch"

    from music.generator import generate_music

    # ---- generation (deployed pipeline, frozen musicgen-small) ----
    trials = []
    t0 = time.time()
    total = len(regions) * args.clips_per_region
    for r in regions:
        for k in range(args.clips_per_region):
            wav = outdir / f"{r['name']}_{k}.wav"
            desc = map_va_to_descriptor(r["valence"], r["arousal"])
            prompt = build_prompt(desc)
            if not wav.exists():
                generate_music(prompt, args.duration, wav)
            trials.append({"region": r["name"], "k": k, "prompt": prompt,
                           "valence": r["valence"], "arousal": r["arousal"],
                           "wav": str(wav)})
            print(f"[{len(trials)}/{total}] {r['name']}_{k} "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    # ---- independent judging via CLAP ----
    print("loading CLAP...", flush=True)
    model, proc = load_clap()
    probe_names = list(PROBES)
    T = embed_text(model, proc, [PROBES[n] for n in probe_names])

    va = {r["name"]: (r["valence"], r["arousal"]) for r in regions}
    for t in trials:
        A = embed_audio(model, proc, t["wav"])
        sims = (A @ T.T).squeeze(0)
        pred = probe_names[int(sims.argmax())]
        t["pred"] = pred
        t["sim_to_target"] = float(sims[probe_names.index(t["region"])])
        t["pred_valence"], t["pred_arousal"] = va[pred]
        t["sims"] = {n: float(sims[i]) for i, n in enumerate(probe_names)}

    n = len(trials)
    top1 = sum(t["pred"] == t["region"] for t in trials) / n
    # sign agreement: does the judged mood land in the same V/A half-plane?
    v_sign = sum((t["pred_valence"] >= 0) == (t["valence"] >= 0) for t in trials) / n
    a_sign = sum((t["pred_arousal"] >= 0) == (t["arousal"] >= 0) for t in trials) / n
    v_mae = float(np.mean([abs(t["pred_valence"] - t["valence"]) for t in trials]))
    a_mae = float(np.mean([abs(t["pred_arousal"] - t["arousal"]) for t in trials]))
    clap = float(np.mean([t["sim_to_target"] for t in trials]))

    res = {"config": {"model": "facebook/musicgen-small (frozen)", "judge": CLAP_ID,
                      "clips_per_region": args.clips_per_region,
                      "duration_s": args.duration, "n_clips": n,
                      "chance_top1": 1.0 / len(probe_names)},
           "metrics": {"top1_accuracy": top1, "valence_sign_agreement": v_sign,
                       "arousal_sign_agreement": a_sign, "valence_mae": v_mae,
                       "arousal_mae": a_mae, "mean_clap_to_target": clap},
           "trials": trials}
    (BASE / "eval" / "results.json").write_text(json.dumps(res, indent=2))

    print(f"\n=== n={n} | chance={1/len(probe_names):.2f} ===")
    print(f"top-1 region accuracy   : {top1:.3f}")
    print(f"valence sign agreement  : {v_sign:.3f}")
    print(f"arousal sign agreement  : {a_sign:.3f}")
    print(f"valence MAE / arousal MAE: {v_mae:.3f} / {a_mae:.3f}")
    print(f"mean CLAP to target probe: {clap:.4f}")


if __name__ == "__main__":
    main()
