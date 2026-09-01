import os, json
import numpy as np
import torch
import torchaudio
import pretty_midi
from einops import rearrange

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model, load_local_trained_model

from stable_audio_tools.data.dataset import midi_to_tokens
from stable_audio_tools.data.midi_utils import save_midi_and_wav
from stable_audio_tools.data.event_codec import Codec
from stable_audio_tools.inference.postprocess import (
    apply_fade_out,
    join_segments,
    leading_silence,
    normalize,
    render_until_valid,
    segment_seconds_total,
    token_stats,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_config_path = "stable_audio_tools/configs/model_configs/txt2audio/stable_audio_3_0.json"

with open(model_config_path) as f:
    model_config = json.load(f)

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]


def render_segment(model, condition, steps, cfg_scale, sigma_min, sigma_max):
    """One draw from the model. Returns the raw (channels, samples) canvas."""
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=[condition],
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )
    return rearrange(output, "b d n -> d (b n)").to(torch.float32)


def process_midi(midi_file, model, codec, sample_rate, output_file, midi_length, prompt,
                 steps=100, cfg_scale=6.0, sigma_min=0.03, sigma_max=500,
                 max_retries=3, peak=0.95, diagnostics_file=None):
    """Render a full MIDI file segment by segment.

    Each segment is generated into a fixed 47.55 s canvas of which only the
    first `seconds_total` seconds carry music; the rest is the zero-padding the
    model was trained against. Segments are trimmed against a measured noise
    floor rather than an absolute threshold, degenerate segments are resampled
    with a new seed instead of being pasted in as silence, and gain is applied
    once over the finished track instead of per segment.
    """
    midi_tokens = midi_to_tokens(midi_file, codec, midi_length=midi_length)
    n_segments = len(midi_tokens)

    final_audio = None
    failures, retried, records = [], [], []

    for i, token in enumerate(midi_tokens):
        stats = token_stats(token, codec)
        seconds_total = segment_seconds_total(stats, midi_length, is_last=(i == n_segments - 1))
        # A segment with no note onsets is a rest or a held sustain: there is no
        # onset energy to detect, so cut on the grid instead of retrying.
        hard_cut = stats["n_onsets"] == 0

        condition = {
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": seconds_total,
            "midi": token,
        }

        segment, diag = render_until_valid(
            lambda attempt: render_segment(model, condition, steps, cfg_scale,
                                           sigma_min, sigma_max),
            sample_rate, seconds_total, max_retries=max_retries, hard_cut=hard_cut,
            on_reject=lambda a, d: print(
                f"[seg {i:4d}/{n_segments}] REJECTED ({d['reason']}), "
                f"resampling {a + 1}/{max_retries}", flush=True))
        attempts = diag["attempts"]

        record = {"segment": i, "seconds_total": seconds_total,
                  "hard_cut": hard_cut, **stats, **diag}
        records.append(record)

        if segment is None:
            failures.append(i)
            print(f"[seg {i:4d}/{n_segments}] FAILED after {max_retries} attempts, skipping",
                  flush=True)
            continue

        if attempts > 1:
            retried.append(i)

        print(f"[seg {i:4d}/{n_segments}] onsets={stats['n_onsets']:4d} "
              f"cond={seconds_total:5.1f}s kept={diag['end_time']:5.2f}s "
              f"snr={diag['snr_db'] if diag['snr_db'] is None else round(diag['snr_db'], 1)} dB "
              f"attempts={attempts}", flush=True)

        # slide this segment back over the previous one's decay by exactly its
        # own leading silence, so a silent head does not become a hole
        overlap = leading_silence(
            segment, sample_rate,
            None if diag.get("music_db") is None else diag["music_db"] - 30.0)
        record["overlap"] = overlap
        final_audio = segment if final_audio is None else join_segments(
            final_audio, segment, sample_rate, overlap)

    if final_audio is None:
        raise RuntimeError("every segment failed, nothing to write")

    # One gain for the whole track: per-segment peak normalisation is what made
    # loudness jump between chunks. Only the very end needs a fade; the
    # junctions are already handled by join_segments.
    final_audio = normalize(apply_fade_out(final_audio, sample_rate, fade_ms=30.0), peak)

    torchaudio.save(output_file, final_audio.mul(32767).to(torch.int16).cpu(), sample_rate)

    duration = final_audio.size(1) / sample_rate
    print(f"\n=== {n_segments} segments, {duration / 60:.2f} min written to {output_file}")
    print(f"=== resampled: {len(retried)} segment(s) {retried}")
    if failures:
        print(f"=== WARNING: {len(failures)} segment(s) missing from the output: {failures}")
    else:
        print("=== no failed segments")

    if diagnostics_file:
        with open(diagnostics_file, "w") as f:
            json.dump({"midi": midi_file, "output": output_file,
                       "n_segments": n_segments, "duration_seconds": duration,
                       "retried": retried, "failed": failures,
                       "settings": {"steps": steps, "cfg_scale": cfg_scale,
                                    "sigma_min": sigma_min, "sigma_max": sigma_max,
                                    "max_retries": max_retries},
                       "segments": records}, f, indent=1)
        print(f"=== diagnostics written to {diagnostics_file}")


if __name__ == "__main__":

    midi_length = 10
    resolution = 100
    codec = Codec(int(midi_length * resolution + 1))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unwarped_path", type=str, required=True)
    parser.add_argument("--midi", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--sigma_min", type=float, default=0.03)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--diagnostics", type=str, default=None,
                        help="optional JSON file with per-segment trim diagnostics")
    args = parser.parse_args()

    prompt = open(args.prompt_file, "r").read()

    model, _ = load_local_trained_model(model_config_path, "", unwarped_path=args.unwarped_path)
    model = model.to(device)

    process_midi(args.midi, model, codec, sample_rate, args.output, midi_length, prompt,
                 steps=args.steps, cfg_scale=args.cfg_scale, sigma_min=args.sigma_min,
                 max_retries=args.max_retries, diagnostics_file=args.diagnostics)
