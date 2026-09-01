"""Export a rendered track segment by segment, as MIDI/audio pairs.

Reconstructs each segment's slice of the concatenated track from the
diagnostics written by full_midi_by_segments.py (or scripts/validate_trim_ab.py)
and writes, per segment, the MIDI that conditioned it and the audio that came
out, plus a CSV of per-segment measurements.

The CSV is the quantitative view of "every chunk sounds different": the model
gets no acoustic context, so level, brightness and stereo width are free to
drift from one segment to the next even though the MIDI is continuous.

    python scripts/export_segments.py --midi FILE --diagnostics diagnostics.json \
        --wav fixed.wav --outdir segments/
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_audio_tools.data.dataset import midi_to_tokens
from stable_audio_tools.data.event_codec import Codec
from stable_audio_tools.data.midi_utils import note_sequence_to_midi_file, tokens_to_notes


def spectral_centroid(x, sr, n_fft=2048):
    if len(x) < n_fft:
        return float("nan")
    hop = n_fft // 2
    n = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    num = den = 0.0
    for i in range(n):
        mag = np.abs(np.fft.rfft(x[i * hop: i * hop + n_fft] * win))
        num += float((mag * freqs).sum())
        den += float(mag.sum())
    return num / den if den > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi", required=True)
    p.add_argument("--diagnostics", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--midi_length", type=int, default=10)
    p.add_argument("--midi_velocity", type=int, default=64,
                   help="velocity to write into the exported MIDI. The codec quantises "
                        "velocity to a single on/off bit, so the tokens carry no dynamics "
                        "and tokens_to_notes reconstructs every note at velocity 1, which "
                        "is inaudible in most synths.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    codec = Codec(int(args.midi_length * 100 + 1))
    with open(args.diagnostics) as f:
        diag = json.load(f)
    segs = diag["segments"]

    tokens = midi_to_tokens(args.midi, codec, midi_length=args.midi_length)
    x, sr = sf.read(args.wav, always_2d=True)

    # Rebuild the segment boundaries. join_segments() slid each segment back
    # over the previous one by its own recorded overlap, so the offsets are
    # exact rather than estimated.
    start = 0.0
    rows = []
    for s in segs:
        i = s["segment"]
        kept = s.get("kept_time") or s["end_time"]
        if kept is None:                       # segment was dropped
            rows.append({"segment": i, "note": "FAILED, not in the track"})
            continue
        if i > 0:
            start -= s.get("overlap", 0.0)
        a, b = int(start * sr), int((start + kept) * sr)
        clip = x[a:b]
        start += kept

        sf.write(os.path.join(args.outdir, f"seg{i:03d}.wav"), clip, sr)
        if i < len(tokens):
            ns = tokens_to_notes(tokens[i], codec)
            for note in ns.notes:
                note.velocity = args.midi_velocity
            note_sequence_to_midi_file(ns, os.path.join(args.outdir, f"seg{i:03d}.mid"))

        mono = clip.mean(axis=1)
        rms = float(np.sqrt((mono ** 2).mean())) if len(mono) else 0.0
        mid = mono[len(mono) // 4: 3 * len(mono) // 4]
        lr = np.abs(clip[:, 0] - clip[:, 1]).mean() if clip.shape[1] == 2 else 0.0
        rows.append({
            "segment": i,
            "start_in_track_s": round(start - kept, 3),
            "kept_s": round(kept, 3),
            "overlap_ms": round(s.get("overlap", 0.0) * 1000, 1),
            "n_onsets": s.get("n_onsets"),
            "seconds_total_cond": s.get("seconds_total"),
            "attempts": s.get("attempts"),
            "snr_db": None if s.get("snr_db") is None else round(s["snr_db"], 1),
            "rms_dbfs": round(20 * np.log10(rms + 1e-12), 2),
            "centroid_hz": round(spectral_centroid(mid, sr), 1),
            "stereo_diff": round(float(lr), 5),
            "note": s.get("reason"),
        })

    csv_path = os.path.join(args.outdir, "segments.csv")
    keys = ["segment", "start_in_track_s", "kept_s", "overlap_ms", "n_onsets",
            "seconds_total_cond", "attempts", "snr_db", "rms_dbfs", "centroid_hz",
            "stereo_diff", "note"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    ok = [r for r in rows if r.get("rms_dbfs") is not None]
    if ok:
        rms = np.array([r["rms_dbfs"] for r in ok])
        cen = np.array([r["centroid_hz"] for r in ok])
        cen = cen[~np.isnan(cen)]
        print(f"=== {len(ok)} segments written to {args.outdir}")
        print(f"    level    p5 {np.percentile(rms, 5):7.2f}  median {np.median(rms):7.2f}  "
              f"p95 {np.percentile(rms, 95):7.2f} dBFS   spread {np.ptp(rms):5.1f} dB")
        print(f"    centroid p5 {np.percentile(cen, 5):7.0f}  median {np.median(cen):7.0f}  "
              f"p95 {np.percentile(cen, 95):7.0f} Hz     spread {np.ptp(cen):5.0f} Hz")
        d_rms = np.abs(np.diff(rms))
        d_cen = np.abs(np.diff(cen))
        print(f"    segment-to-segment jump: level median {np.median(d_rms):.2f} dB "
              f"(max {d_rms.max():.2f}), centroid median {np.median(d_cen):.0f} Hz "
              f"(max {d_cen.max():.0f})")
    print(f"=== {csv_path}")


if __name__ == "__main__":
    main()
