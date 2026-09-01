"""Post-processing for segment-wise full-track rendering.

The model always generates a fixed-size canvas (``sample_size``, 47.55 s at
44.1 kHz), of which only the first ``seconds_total`` seconds are meant to carry
music. Every training sample was a zero-padded buffer with the audio glued to
the front (see ``PadCrop_Normalized_T`` in ``../data/utils.py``), so:

* music always starts at t = 0, therefore the head must never be trimmed;
* the tail of the canvas is always padding, therefore it must be trimmed, and
  it doubles as a free per-segment measurement of the model's noise floor.

The previous ``remove_silence`` helper used an absolute energy threshold on the
peak-normalised segment, trimmed both ends, and returned the *whole* canvas
when nothing crossed the threshold. That last behaviour is how a segment that
collapsed to the silence mode turned into ~47 s of silence pasted into the
middle of a track. Here a failed detection is reported to the caller instead,
so it can resample with a different seed.
"""

import math
import typing as tp

import torch


def frame_rms_db(audio: torch.Tensor, sample_rate: int, hop_ms: float = 20.0):
    """Per-frame RMS in dBFS of a (channels, samples) tensor. Returns (frames, hop)."""
    hop = max(1, int(hop_ms * 1e-3 * sample_rate))
    mono = audio.float().mean(dim=0)
    n_frames = mono.numel() // hop
    if n_frames < 1:
        return torch.empty(0, device=audio.device), hop
    frames = mono[: n_frames * hop].reshape(n_frames, hop)
    return 20 * torch.log10(frames.pow(2).mean(dim=1).sqrt() + 1e-12), hop


def find_music_end(
    audio: torch.Tensor,
    sample_rate: int,
    seconds_total: float,
    hop_ms: float = 20.0,
    snr_min_db: float = 12.0,
    rel_db: float = 6.0,
    below_music_db: float = 30.0,
    floor_window: float = 5.0,
    search_back: float = 1.5,
    search_fwd: float = 2.5,
) -> dict:
    """Locate the end of the music inside a generated canvas.

    Must be called on the *raw* model output, before any normalisation: the
    absolute levels are what make the noise floor measurable.

    Returns a diagnostics dict. ``ok`` is False when the segment is degenerate
    (collapsed to the silence mode, or noise all the way through), in which case
    the caller should resample with a new seed rather than use the audio.
    """
    diag = {
        "ok": False,
        "reason": None,
        "end_sample": None,
        "end_time": None,
        "floor_db": None,
        "music_db": None,
        "snr_db": None,
        "thresh_db": None,
    }

    n_samples = audio.size(1)
    canvas_seconds = n_samples / sample_rate
    rms_db, hop = frame_rms_db(audio, sample_rate, hop_ms)
    n_frames = rms_db.numel()
    if n_frames < 10:
        diag["reason"] = "canvas too short to analyse"
        return diag

    def frame_of(t: float) -> int:
        return int(max(0.0, t) * sample_rate / hop)

    # Noise floor: the tail of the canvas is padding under every condition we
    # ever generate with (seconds_total <= 20 against a 47.55 s canvas).
    floor_start = frame_of(canvas_seconds - floor_window)
    tail = rms_db[floor_start:]
    if tail.numel() < 5:
        diag["reason"] = "no padding region to measure the noise floor"
        return diag
    floor_db = tail.median().item()

    # Music level: the part of the segment that should be unambiguously music.
    body = rms_db[: max(frame_of(seconds_total * 0.8), 5)]
    music_db = body.median().item()

    snr_db = music_db - floor_db
    diag.update(floor_db=floor_db, music_db=music_db, snr_db=snr_db)

    if snr_db < snr_min_db:
        diag["reason"] = f"degenerate segment: SNR {snr_db:.1f} dB < {snr_min_db:.1f} dB"
        return diag

    # Search for the boundary only near where the timing condition says it is.
    # The threshold is musical, not floor-relative: a tail sitting 30 dB under
    # the passage is inaudible once the next segment enters, and keeping it
    # (the floor is another 40 dB down) pastes up to half a second of near
    # silence onto the end of every segment.
    thresh_db = max(floor_db + rel_db, music_db - below_music_db)
    diag["thresh_db"] = thresh_db
    lo = frame_of(seconds_total - search_back)
    hi = min(frame_of(seconds_total + search_fwd), n_frames)
    if hi <= lo:
        diag["reason"] = "empty search window"
        return diag

    above = (rms_db[lo:hi] > thresh_db).nonzero(as_tuple=True)[0]
    if len(above):
        end_frame = lo + above[-1].item() + 1
    else:
        # Nothing above the floor in the window: the model stopped early. Fall
        # back to the last frame above threshold before the window.
        head = (rms_db[:lo] > thresh_db).nonzero(as_tuple=True)[0]
        if not len(head):
            diag["reason"] = "no music found before the search window"
            return diag
        end_frame = head[-1].item() + 1

    end_sample = min(end_frame * hop, n_samples)
    diag.update(ok=True, reason="ok", end_sample=end_sample,
                end_time=end_sample / sample_rate)
    return diag


def apply_fade_out(audio: torch.Tensor, sample_rate: int, fade_ms: float = 30.0):
    """In-place-ish fade at the tail so a hard cut does not click."""
    n = min(int(fade_ms * 1e-3 * sample_rate), audio.size(1))
    if n > 0:
        audio = audio.clone()
        audio[:, -n:] *= torch.linspace(1, 0, n, device=audio.device)
    return audio


def trim_to_music(
    audio: torch.Tensor,
    sample_rate: int,
    seconds_total: float,
    fade_ms: float = 0.0,
    hard_cut: bool = False,
    tail_margin_ms: float = 0.0,
    **kwargs,
) -> tp.Tuple[tp.Optional[torch.Tensor], dict]:
    """Trim a generated canvas down to its music.

    ``hard_cut=True`` skips detection and cuts at exactly ``seconds_total``. Use
    it for segments whose MIDI holds no note onsets (a rest or a held sustain),
    where there is no onset energy to detect and retrying would be pointless.

    ``fade_ms`` defaults to 0 because segments are joined with ``crossfade_audio``,
    which already fades the tail. Fading here as well drives the junction to
    silence for the length of the fade, which shows up as a hole in the envelope
    at every segment boundary. Only the very end of the finished track needs an
    explicit fade.

    ``tail_margin_ms`` keeps that much of the canvas past the detected boundary.
    It defaults to 0 and is rarely useful: everything past the boundary is by
    definition below the detection threshold, so crossfading over a margin only
    guarantees a quiet join. Use ``join_segments`` instead, which overlaps the
    next segment onto this one's real decay.

    Returns ``(audio, diag)`` with ``audio=None`` when the segment is degenerate.
    """
    margin = int(tail_margin_ms * 1e-3 * sample_rate)

    if hard_cut:
        end = min(int(seconds_total * sample_rate), audio.size(1))
        diag = {"ok": True, "reason": "hard cut (no note onsets)",
                "end_sample": end, "end_time": end / sample_rate,
                "floor_db": None, "music_db": None, "snr_db": None,
                "thresh_db": None}
    else:
        diag = find_music_end(audio, sample_rate, seconds_total, **kwargs)
        if not diag["ok"]:
            return None, diag
        end = diag["end_sample"]

    kept = min(end + margin, audio.size(1))
    diag["kept_sample"] = kept
    diag["kept_time"] = kept / sample_rate
    return apply_fade_out(audio[:, :kept], sample_rate, fade_ms), diag


def crossfade_audio(audio1: torch.Tensor, audio2: torch.Tensor,
                    fade_duration: float, sample_rate: int,
                    equal_power: bool = True) -> torch.Tensor:
    """Crossfade two segments. All three slices use the same clamped length.

    Consecutive segments are independent draws, so their overlap is
    uncorrelated and an equal-power (sqrt) fade holds the level constant
    through the junction; a linear fade would dip by 3 dB.
    """
    n = min(int(fade_duration * sample_rate), audio1.size(1), audio2.size(1))
    if n <= 0:
        return torch.cat([audio1, audio2], dim=1)
    ramp = torch.linspace(1, 0, n, device=audio1.device)
    fade_out, fade_in = (ramp.sqrt(), (1 - ramp).sqrt()) if equal_power else (ramp, 1 - ramp)
    crossfaded = audio1[:, -n:] * fade_out + audio2[:, :n] * fade_in
    return torch.cat([audio1[:, :-n], crossfaded, audio2[:, n:]], dim=1)


def leading_silence(audio: torch.Tensor, sample_rate: int,
                    thresh_db: tp.Optional[float] = None,
                    below_music_db: float = 30.0,
                    hop_ms: float = 5.0, max_seconds: float = 0.5) -> float:
    """Length in seconds of the run of silence at the head of a segment.

    The model has no acoustic context, so it renders every segment from digital
    silence. When a segment's MIDI opens on a short rest, its head is silent,
    and butting it against the previous segment leaves an audible hole where a
    real performance would have carried the previous notes' decay across.

    The threshold is relative to the segment's own music level, not to the noise
    floor: a head sitting 40 dB under the music is a hole to the ear even though
    it is well above the floor.
    """
    db, hop = frame_rms_db(audio, sample_rate, hop_ms)
    if db.numel() < 2:
        return 0.0
    if thresh_db is None:
        thresh_db = db.median().item() - below_music_db
    above = (db > thresh_db).nonzero(as_tuple=True)[0]
    if not len(above):
        return 0.0
    return min(above[0].item() * hop / sample_rate, max_seconds)


def join_segments(a: torch.Tensor, b: torch.Tensor, sample_rate: int,
                  overlap_seconds: float, tail_fade_ms: float = 0.0) -> torch.Tensor:
    """Append `b` to `a`, sliding it back over `a`'s decay by `overlap_seconds`.

    Unlike a crossfade this does not fade `a` out: `b`'s head is silent over the
    overlap by construction, so `a`'s decay simply continues underneath until
    `b` enters. Nothing is removed from either segment, and two different
    passages are never heard on top of each other.

    `tail_fade_ms` defaults to 0: the previous segment is cut at a point that is
    already ~45 dB below its own music, so the discontinuity is inaudible, while
    a fade would drive the join to digital silence and create the very hole this
    is meant to avoid.
    """
    a = apply_fade_out(a, sample_rate, tail_fade_ms)
    n = min(int(overlap_seconds * sample_rate), a.size(1), b.size(1))
    if n <= 0:
        return torch.cat([a, b], dim=1)
    out = a.new_zeros(a.size(0), a.size(1) + b.size(1) - n)
    out[:, : a.size(1)] = a
    gain = torch.ones(b.size(1), device=b.device)
    gain[:n] = torch.linspace(0, 1, n, device=b.device).sqrt()
    out[:, a.size(1) - n:] += b * gain
    return out


# --- MIDI token introspection ------------------------------------------------

def token_stats(tokens: torch.Tensor, codec, resolution: int = 100) -> dict:
    """Note-onset count and content span (seconds) of one tokenised segment.

    Token ids are read off the codec's own mapping rather than hardcoded, so
    this stays correct if the vocabulary layout changes. Shift events are the
    first block of the vocabulary and are encoded as identity (``{t: t}`` in
    ``Codec.__init__``), so a shift id is also its frame index.
    """
    shift_ids = codec.encoding_mapping["shift"]
    velocity_on = codec.encoding_mapping["velocity"][1]
    max_shift_id = max(shift_ids)

    t = tokens.flatten().to(torch.long)
    shifts = t[t <= max_shift_id]
    return {
        "n_onsets": int((t == velocity_on).sum().item()),
        "midi_span": float(shifts.max().item() / resolution) if shifts.numel() else 0.0,
        "n_tokens": int(t.numel()),
    }


def segment_seconds_total(stats: dict, midi_length: float, is_last: bool,
                          min_seconds: float = 1.0, max_seconds: float = 20.0) -> float:
    """Timing condition for a segment.

    Interior segments span exactly ``midi_length`` of MIDI by construction (the
    tokeniser cuts on a fixed absolute-time grid), so that is what the model
    should be asked to fill. Only the final, partial segment needs its length
    derived from the tokens; the previous hardcoded 17 s asked the model to fill
    time the MIDI did not contain, which invites silence.
    """
    if not is_last:
        return float(midi_length)
    span = math.ceil(stats["midi_span"]) + 1.0  # + release tail
    return float(min(max(span, min_seconds), max_seconds))
