"""Silence report for a rendered full track.

Finds runs of near-silence that are longer than a musical pause is likely to
be, and plots the waveform and the level envelope so they can be read off.
Give it one wav, or two (fixed + legacy) to compare.

    python scripts/plot_silence_report.py fixed.wav legacy.wav \
        --diagnostics diagnostics.json --out report.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def envelope_db(x, sr, hop_ms=50.0):
    hop = max(1, int(hop_ms * 1e-3 * sr))
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-12), hop


def silence_runs(db, hop, sr, thresh_db, min_seconds=1.5):
    """Contiguous runs below thresh_db lasting at least min_seconds."""
    quiet = db < thresh_db
    runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(quiet)))
    out = []
    for a, b in runs:
        dur = (b - a) * hop / sr
        if dur >= min_seconds:
            out.append({"start_s": a * hop / sr, "end_s": b * hop / sr, "seconds": dur})
    return out


def analyse(path, hop_ms=50.0, rel_db=35.0, min_seconds=1.5):
    x, sr = sf.read(path, always_2d=True)
    mono = x.mean(axis=1)
    db, hop = envelope_db(mono, sr, hop_ms)
    active = db[db > db.max() - 60]
    ref = np.median(active) if len(active) else db.max()
    thresh = ref - rel_db
    return {"path": path, "sr": sr, "mono": mono, "db": db, "hop": hop,
            "duration": len(mono) / sr, "ref_db": ref, "thresh_db": thresh,
            "runs": silence_runs(db, hop, sr, thresh, min_seconds)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wavs", nargs="+")
    p.add_argument("--diagnostics", default=None)
    p.add_argument("--out", default="silence_report.png")
    p.add_argument("--rel_db", type=float, default=35.0)
    p.add_argument("--min_seconds", type=float, default=1.5)
    args = p.parse_args()

    results = [analyse(w, rel_db=args.rel_db, min_seconds=args.min_seconds)
               for w in args.wavs]

    diag = None
    if args.diagnostics:
        with open(args.diagnostics) as f:
            diag = json.load(f)

    for r in results:
        print(f"\n=== {r['path']}")
        print(f"    {r['duration'] / 60:.2f} min, reference level {r['ref_db']:.1f} dBFS, "
              f"silence threshold {r['thresh_db']:.1f} dBFS")
        if not r["runs"]:
            print(f"    no silent run >= {args.min_seconds}s")
        else:
            total = sum(x["seconds"] for x in r["runs"])
            print(f"    {len(r['runs'])} silent run(s) >= {args.min_seconds}s, {total:.1f}s total:")
            for x in sorted(r["runs"], key=lambda d: -d["seconds"])[:15]:
                m, s = divmod(x["start_s"], 60)
                print(f"      {int(m):3d}:{s:05.2f}  {x['seconds']:6.2f}s")

    n = len(results)
    fig, axes = plt.subplots(2 * n, 1, figsize=(16, 4.2 * n), squeeze=False)
    for k, r in enumerate(results):
        t = np.arange(len(r["mono"])) / r["sr"]
        ax = axes[2 * k][0]
        step = max(1, len(t) // 300000)
        ax.plot(t[::step] / 60, r["mono"][::step], lw=0.25, color="#2a6fb0")
        ax.set_title(f"{r['path'].split('/')[-1]}  waveform  "
                     f"({r['duration'] / 60:.2f} min, {len(r['runs'])} silent run(s) "
                     f">= {args.min_seconds}s)")
        ax.set_ylabel("amplitude")
        ax.set_xlim(0, r["duration"] / 60)

        axd = axes[2 * k + 1][0]
        td = np.arange(len(r["db"])) * r["hop"] / r["sr"] / 60
        axd.plot(td, r["db"], lw=0.5, color="#333333")
        axd.axhline(r["thresh_db"], color="crimson", ls="--", lw=0.9,
                    label=f"silence threshold {r['thresh_db']:.0f} dBFS")
        for x in r["runs"]:
            for a in (ax, axd):
                a.axvspan(x["start_s"] / 60, x["end_s"] / 60, color="crimson", alpha=0.30, lw=0)
        axd.set_ylabel("RMS (dBFS)")
        axd.set_xlabel("minutes")
        axd.set_xlim(0, r["duration"] / 60)
        axd.set_ylim(max(-100, r["db"].min() - 3), 3)
        axd.legend(loc="lower right", fontsize=8)

        if diag and k == 0:
            for s in diag["segments"]:
                if s.get("attempts", 1) > 1 and s.get("ok"):
                    pass  # marked in the console table instead; positions shift after trimming

    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"\n=== plot written to {args.out}")

    if diag:
        segs = diag["segments"]
        kept = [s["end_time"] for s in segs if s.get("end_time")]
        leg = [s["legacy_seconds"] for s in segs if s.get("legacy_seconds") is not None]
        whole = [s["segment"] for s in segs
                 if s.get("legacy_note", "") and "whole canvas" in s["legacy_note"]]
        tiny = [s["segment"] for s in segs
                if s.get("legacy_seconds") is not None and s["legacy_seconds"] < 1.0]
        print(f"\n=== diagnostics: {diag['n_segments']} segments")
        print(f"    resampled by the fix : {len(diag['retried'])} {diag['retried']}")
        print(f"    failed after retries : {len(diag['failed'])} {diag['failed']}")
        print(f"    legacy crossfade crash: {len(diag.get('legacy_crashes', []))} "
              f"{diag.get('legacy_crashes', [])}")
        print(f"    legacy kept whole canvas: {len(whole)} {whole[:20]}")
        print(f"    legacy kept < 1s        : {len(tiny)} {tiny[:20]}")
        if kept:
            print(f"    fixed kept  min/median/max: {min(kept):.2f} / "
                  f"{float(np.median(kept)):.2f} / {max(kept):.2f} s")
        if leg:
            print(f"    legacy kept min/median/max: {min(leg):.2f} / "
                  f"{float(np.median(leg)):.2f} / {max(leg):.2f} s")
        snrs = [s["snr_db"] for s in segs if s.get("snr_db") is not None]
        if snrs:
            print(f"    measured SNR min/median/max: {min(snrs):.1f} / "
                  f"{float(np.median(snrs)):.1f} / {max(snrs):.1f} dB")


if __name__ == "__main__":
    main()
