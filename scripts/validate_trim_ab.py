"""A/B validation of the segment trimming fix on a full-length track.

Generates every segment once and runs both post-processing paths over the same
draws, so the comparison is not confounded by sampling noise:

* legacy  -- per-segment peak normalisation + absolute-energy `remove_silence`,
             both ends trimmed, whole canvas returned on detection failure,
             and the off-by-fade concatenation bug. No retries (the old code
             had none), so it always uses attempt 0.
* fixed   -- noise-floor-relative trim of the tail only, degenerate segments
             resampled with a new seed, one gain over the finished track.

Writes both wavs plus a JSON with per-segment diagnostics for plotting.

Usage:
    python scripts/validate_trim_ab.py --unwarped_path CKPT --midi FILE \
        --prompt_file FILE --outdir DIR
"""

import argparse
import json
import os
import sys

import torch
import torchaudio
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_audio_tools.data.dataset import midi_to_tokens
from stable_audio_tools.data.event_codec import Codec
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.inference.postprocess import (
    apply_fade_out,
    join_segments,
    leading_silence,
    normalize,
    frame_rms_db,
    segment_seconds_total,
    token_stats,
    trim_to_music,
)
from stable_audio_tools.models.pretrained import load_local_trained_model


# --- the original post-processing, reproduced verbatim for comparison --------

def legacy_remove_silence(audio, threshold=0.01):
    energy = audio.pow(2).mean(dim=0)
    non_silent_indices = (energy > threshold).nonzero(as_tuple=True)[0]
    if len(non_silent_indices) == 0:
        return audio, "returned whole canvas"
    start, end = non_silent_indices[0], non_silent_indices[-1]
    return audio[:, start:end], f"trimmed [{start.item()}:{end.item()}]"


def legacy_crossfade(audio1, audio2, fade_duration, sample_rate):
    """Verbatim original. Raises RuntimeError when either side is shorter than
    the fade, because `fade_out` has length `fade_samples` while the slice it
    multiplies has length `fade_samples_`."""
    fade_samples = int(fade_duration * sample_rate)
    fade_out = torch.linspace(1, 0, fade_samples).to(audio1.device)
    fade_in = torch.linspace(0, 1, fade_samples).to(audio1.device)
    fade_samples_ = min(fade_samples, audio1.size(1), audio2.size(1))
    crossfaded = audio1[:, -fade_samples_:] * fade_out + audio2[:, :fade_samples_] * fade_in
    return torch.cat([audio1[:, :-fade_samples], crossfaded, audio2[:, fade_samples:]], dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unwarped_path", required=True)
    p.add_argument("--midi", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--config", default="stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0.json")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=6.0)
    p.add_argument("--sigma_min", type=float, default=0.03)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--midi_length", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="only render the first N segments")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config) as f:
        model_config = json.load(f)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    os.makedirs(args.outdir, exist_ok=True)
    codec = Codec(int(args.midi_length * 100 + 1))
    prompt = open(args.prompt_file).read()

    model, _ = load_local_trained_model(args.config, "", unwarped_path=args.unwarped_path)
    model = model.to(device)

    midi_tokens = midi_to_tokens(args.midi, codec, midi_length=args.midi_length)
    n_segments = len(midi_tokens)
    if args.limit:
        midi_tokens = midi_tokens[: args.limit]
    print(f"=== {args.midi}: {n_segments} segments, rendering {len(midi_tokens)}", flush=True)

    fixed_audio, legacy_audio = None, None
    records, retried, failed, legacy_crashes = [], [], [], []

    for i, token in enumerate(midi_tokens):
        is_last = i == n_segments - 1
        stats = token_stats(token, codec)
        seconds_total = segment_seconds_total(stats, args.midi_length, is_last=is_last)
        hard_cut = stats["n_onsets"] == 0

        condition = {"prompt": prompt, "seconds_start": 0,
                     "seconds_total": seconds_total, "midi": token}

        segment, diag, attempts, legacy_note, legacy_len = None, None, 0, None, None
        for attempt in range(args.max_retries):
            attempts = attempt + 1
            out = generate_diffusion_cond(
                model, steps=args.steps, cfg_scale=args.cfg_scale,
                conditioning=[condition], sample_size=sample_size,
                sigma_min=args.sigma_min, sigma_max=500,
                sampler_type="dpmpp-3m-sde", device=device)
            canvas = rearrange(out, "b d n -> d (b n)").to(torch.float32)

            if attempt == 0:
                # what the old pipeline would have produced from this same draw
                norm = (canvas / torch.max(torch.abs(canvas)).clamp(min=1e-12)).clamp(-1, 1)
                leg, legacy_note = legacy_remove_silence(norm)
                legacy_len = leg.size(1) / sample_rate
                if legacy_audio is None:
                    legacy_audio = leg
                else:
                    try:
                        legacy_audio = legacy_crossfade(legacy_audio, leg, 0.05, sample_rate)
                    except RuntimeError as e:
                        # the old code dies here; record it and keep going so the
                        # rest of the comparison still gets produced
                        legacy_note += f" | CRASH in crossfade: {e}"
                        legacy_crashes.append(i)
                        legacy_audio = torch.cat([legacy_audio, leg], dim=1)

            segment, diag = trim_to_music(canvas, sample_rate, seconds_total, hard_cut=hard_cut)
            if segment is not None:
                break
            print(f"[seg {i:4d}] REJECTED ({diag['reason']}), resample {attempt + 1}", flush=True)

        # keep a coarse envelope of the accepted segment for plotting
        env = None
        if segment is not None:
            db, hop = frame_rms_db(segment, sample_rate, hop_ms=100.0)
            env = [round(v, 2) for v in db.tolist()]

        records.append({"segment": i, "seconds_total": seconds_total, "attempts": attempts,
                        "hard_cut": hard_cut, "is_last": is_last,
                        "legacy_note": legacy_note, "legacy_seconds": legacy_len,
                        "fixed_envelope_db_100ms": env, **stats, **diag})

        if segment is None:
            failed.append(i)
            print(f"[seg {i:4d}] FAILED after {args.max_retries} attempts", flush=True)
            continue
        if attempts > 1:
            retried.append(i)

        snr = "n/a" if diag["snr_db"] is None else f"{diag['snr_db']:5.1f}"
        print(f"[seg {i:4d}/{n_segments}] onsets={stats['n_onsets']:4d} "
              f"cond={seconds_total:5.1f}s fixed={diag['end_time']:6.2f}s "
              f"legacy={legacy_len:6.2f}s snr={snr} dB att={attempts}", flush=True)

        overlap = leading_silence(
            segment, sample_rate,
            None if diag.get("music_db") is None else diag["music_db"] - 30.0)
        records[-1]["overlap"] = overlap
        fixed_audio = segment if fixed_audio is None else join_segments(
            fixed_audio, segment, sample_rate, overlap)

    def save(audio, name):
        audio = normalize(apply_fade_out(audio, sample_rate, fade_ms=30.0))
        path = os.path.join(args.outdir, name)
        torchaudio.save(path, audio.mul(32767).to(torch.int16).cpu(), sample_rate)
        print(f"=== wrote {path}  {audio.size(1) / sample_rate / 60:.2f} min")
        return path

    fixed_path = save(fixed_audio, "fixed.wav")
    legacy_path = save(legacy_audio, "legacy.wav")

    summary = {"midi": args.midi, "n_segments": n_segments,
               "rendered": len(midi_tokens), "retried": retried, "failed": failed,
               "legacy_crashes": legacy_crashes,
               "sample_rate": sample_rate,
               "fixed_wav": fixed_path, "legacy_wav": legacy_path,
               "settings": vars(args), "segments": records}
    with open(os.path.join(args.outdir, "diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\n=== retried {len(retried)} segment(s): {retried}")
    print(f"=== failed  {len(failed)} segment(s): {failed}")
    print(f"=== legacy pipeline would have crashed at {len(legacy_crashes)} segment(s): {legacy_crashes}")


if __name__ == "__main__":
    main()
