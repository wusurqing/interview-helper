"""Nonblocking capture subscription and bounded, memory-only Opus encoding."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import logging
import queue
import threading
import time
from uuid import uuid4

import numpy as np

RATE = 16000
BLOCK_SAMPLES = RATE * 2
ENCODE_AT = RATE * 8 // 5  # leave room for the next device callback under the 2 s cap
SPEAKERS = ("interviewer", "candidate")
LATE_FINAL_SECONDS = 5
QUIET_RMS = 0.0015
VOICE_RMS = 0.006
_log = logging.getLogger(__name__)


class CapturedAudio(np.ndarray):
    """An ndarray view which keeps source coordinates through queue slicing."""

    def __new__(cls, samples, capture_id, sample_start, anchor, *, discontinuity=False):
        value = np.asarray(samples).view(cls)
        value.capture_id = capture_id
        value.sample_start = sample_start
        value.capture_anchor = anchor
        value.discontinuity = discontinuity
        return value

    def __array_finalize__(self, parent):
        self.capture_id = getattr(parent, "capture_id", "")
        self.sample_start = getattr(parent, "sample_start", 0)
        self.capture_anchor = getattr(parent, "capture_anchor", 0.0)
        self.discontinuity = getattr(parent, "discontinuity", False)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(value, CapturedAudio) and isinstance(item, slice):
            start, _, step = item.indices(len(self))
            if step == 1:
                value.sample_start = self.sample_start + start
                value.discontinuity = self.discontinuity and start == 0
        return value

    @property
    def started_mono(self):
        return self.capture_anchor + self.sample_start / RATE


def encode_opus(pcm: np.ndarray) -> bytes:
    import av  # warm on the encoding thread, never in a capture callback

    target = io.BytesIO()
    with av.open(target, mode="w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=RATE)
        stream.layout = "mono"
        stream.bit_rate = 24000
        stream.codec_context.options = {"application": "voip"}
        frame = av.AudioFrame.from_ndarray(np.asarray(pcm, dtype=np.int16).reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RATE
        frame.time_base = Fraction(1, RATE)
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return target.getvalue()


@dataclass(frozen=True)
class Clip:
    asset_id: str
    capture_id: str
    speaker: str
    sample_start: int
    sample_count: int
    anchor: float
    payload: bytes

    @property
    def started_mono(self):
        return self.anchor + self.sample_start / RATE


@dataclass(frozen=True)
class CaptureRange:
    """One uninterrupted, role-owned portion of a physical capture."""

    id: str
    source: str
    speaker: str
    anchor: float
    start: int
    end: int

    def clock(self, sample):
        return self.anchor + sample / RATE

    def sample(self, at):
        return round((at - self.anchor) * RATE)


def _merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _candidate_tail_end(capture, regions, previous, boundary):
    if previous is None:
        return boundary
    if previous["alignment"] == "capture":
        if previous.get("source_capture_id") == capture.source:
            return max(boundary, previous.get("audio_ended_mono") or boundary)
        return boundary
    if previous.get("capture_id") != capture.id:
        return boundary
    limit = previous["observed_end"] if previous["is_final"] else float("inf")
    cut = capture.sample(boundary)
    end = max((capture.clock(b) for a, b, _ in regions if a < cut < b), default=boundary)
    return max(boundary, min(end, limit))


def _select_audio_ranges(speaker, utterances, captures, speech, losses, start, end, timed_out, previous_candidate):
    tracks = [capture for capture in captures.values() if capture.speaker == speaker]
    floors = {
        c.id: _candidate_tail_end(c, speech[c.id], previous_candidate, start) if speaker == "candidate" else start
        for c in tracks
    }
    selected: dict[str, list[tuple[int, int]]] = {}
    approximate = False
    reason = None

    def include(capture, first, last):
        left = max(capture.start, capture.sample(first))
        right = min(capture.end, capture.sample(last))
        if left < right:
            selected.setdefault(capture.id, []).append((left, right))
            return right - left
        return 0

    parsed = sorted(utterances, key=lambda item: item["started_mono"])
    for index, utterance in enumerate(parsed):
        observed = utterance["started_mono"]
        if utterance["alignment"] == "capture":
            audio_end = utterance.get("audio_ended_mono")
            if audio_end is None:
                continue
            source = utterance.get("source_capture_id")
            first = utterance["started_mono"]
            available = sum(include(c, first, audio_end) for c in tracks if not source or c.source == source)
            if available + 1 < round((audio_end - first) * RATE):
                reason = reason or "samples_missing"
            continue

        capture_id = utterance.get("capture_id")
        capture = captures.get(capture_id) if capture_id else AudioRecorder._capture_at(tracks, speaker, observed)
        following = parsed[index + 1]["started_mono"] if index + 1 < len(parsed) else float("inf")
        final_limit = min(following, utterance["observed_end"])
        upper = min(end, final_limit)
        floor = floors.get(capture.id, start) if capture is not None else start
        if index:
            floor = max(floor, parsed[index - 1]["started_mono"])
        lower = max(floor, observed - 2.0)
        if capture is None and not capture_id:
            # If the cloud was very late, a single eligible capture still
            # provides role-level audio. Never substitute for a pinned capture.
            eligible = [c for c in tracks if c.clock(c.start) < upper and
                        c.clock(c.start) <= observed + 0.25 and c.clock(c.end) > floor]
            if len(eligible) == 1 and include(eligible[0], max(floor, floors[eligible[0].id]), upper):
                approximate = True
                continue
        if capture is None or capture.speaker != speaker:
            reason = reason or "unmatched"
            continue
        left, right = capture.sample(lower), capture.sample(upper)
        voice = [(a, b, clear) for a, b, clear in speech[capture.id] if a < right and b > left]
        leading = [i for i, (a, b, _) in enumerate(voice)
                   if a <= capture.sample(observed + 0.25) and b >= capture.sample(observed - 1.0)]
        if leading:
            # Keep a continuously voiced candidate tail across the next
            # question. A new region after that boundary belongs elsewhere.
            voice_end = capture.clock(voice[-1][1])
            if speaker == "candidate" and upper == end and voice_end > end:
                upper = min(final_limit, voice_end)
            include(capture, max(floor, capture.clock(voice[leading[-1]][0])),
                    min(upper, voice_end))
            approximate |= voice[leading[-1]][0] < left or not all(clear for _, _, clear in voice[leading[-1]:])
        elif include(capture, floor, upper):
            # The final itself is evidence of speech even when the archive VAD
            # sees nothing. Retain only real PCM from this role and round.
            approximate = True
        else:
            reason = reason or "unmatched"

    tail_start = max((captures[key].clock(right) for key, spans in selected.items() for _, right in spans), default=start)
    for capture in tracks:
        left, right = capture.sample(floors[capture.id]), capture.sample(end)
        if not (capture.start < right and capture.end > left):
            continue
        if timed_out:
            approximate |= bool(include(capture, max(floors[capture.id], tail_start), end))
        else:
            # A disconnect can end a captured utterance before the provider
            # even announces it. Keep voiced material after recognized speech.
            for a, b, _ in speech[capture.id]:
                if max(left, capture.sample(tail_start)) <= a < right:
                    approximate |= bool(include(capture, capture.clock(a), min(end, capture.clock(b))))
        for a, b, failure in losses.get(capture.id, ()):
            if a < right and b > left:
                include(capture, capture.clock(max(a, left)), capture.clock(min(b, right)))
                reason = failure
    return selected, approximate, reason


class AudioRecorder:
    def __init__(self, *, budget=64 * 1024 * 1024):
        self._budget = budget
        self._queue = queue.Queue(maxsize=128)
        self._lock = threading.RLock()
        self._progress = threading.Condition(self._lock)
        self._accepted = self._processed = self._encoded = 0
        self._checkpoints: deque[int] = deque()
        self._sealed = self._shutdown = False
        self._buffered = {}
        self._pending_samples = dict.fromkeys(SPEAKERS, 0)
        self._bytes = 0
        self._clips: dict[str, Clip] = {}
        self._captures: dict[str, CaptureRange] = {}
        self._capture_ticket: dict[str, int] = {}
        self._active: dict[str, str] = {}
        self._regions: dict[str, list[tuple[int, int, bool]]] = {}
        self._speech: dict[str, tuple[int, int, bool]] = {}
        self._losses: dict[str, list[tuple[int, int, str]]] = {}
        self._worker_capture: dict[str, str] = {}
        self._worker_end: dict[str, int] = {}
        self._voice_floor: dict[str, int] = {}
        self._fallback = {}
        self._thread = threading.Thread(target=self._run, name="archive.opus", daemon=True)
        self._thread.start()

    @property
    def memory_bytes(self):
        with self._lock:
            return self._bytes

    def offer(self, samples, speaker):
        if speaker not in self._pending_samples or not len(samples):
            return
        count = len(samples)
        # The callback does only a bounded copy/enqueue. No codec, file IO or VAD.
        with self._lock:
            if self._sealed:
                return
            if getattr(samples, "capture_id", ""):
                source, start, anchor = samples.capture_id, samples.sample_start, samples.capture_anchor
            else:
                if speaker not in self._fallback:
                    self._fallback[speaker] = (uuid4().hex, 0, time.monotonic() - count / RATE)
                source, start, anchor = self._fallback[speaker]
                self._fallback[speaker] = source, start + count, anchor
            previous = self._captures.get(self._active.get(speaker))
            if previous and previous.source == source and previous.end == start:
                capture_id, capture_start = previous.id, previous.start
            else:
                capture_id, capture_start = uuid4().hex, start
                self._active[speaker] = capture_id
            end = start + count
            self._captures[capture_id] = CaptureRange(capture_id, source, speaker, anchor, capture_start, end)
            if getattr(samples, "discontinuity", False):
                # Hardware reports a break without reporting how many samples
                # were lost. Tag the first real sample after this boundary.
                self._record_loss(capture_id, start, start + 1, "capture_gap")
            if self._pending_samples[speaker] + count > BLOCK_SAMPLES or self._bytes + samples.nbytes > self._budget:
                self._record_loss(capture_id, start, end, "buffer_full")
                return
            block = (self._accepted + 1, capture_id, start, np.array(samples, copy=True, subok=False))
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                self._record_loss(capture_id, start, end, "queue_full")
                return
            self._accepted += 1
            self._capture_ticket[capture_id] = self._accepted
            self._pending_samples[speaker] += count
            self._bytes += block[-1].nbytes

    def end_capture(self, speaker):
        with self._lock:
            self._active.pop(speaker, None)
            self._fallback.pop(speaker, None)

    def _record_loss(self, capture_id, start, end, reason):
        losses = self._losses.setdefault(capture_id, [])
        if losses and losses[-1][1] == start and losses[-1][2] == reason:
            losses[-1] = (losses[-1][0], end, reason)
        else:
            losses.append((start, end, reason))

    def _encode(self, capture_id, force=False):
        state = self._buffered.get(capture_id)
        if not state:
            return
        pcm, start = state
        if not force and len(pcm) < ENCODE_AT:
            return
        del self._buffered[capture_id]
        try:
            payload = encode_opus(pcm)
        except Exception:
            _log.exception("Opus encode failed capture=%s samples=%s:%s", capture_id, start, start + len(pcm))
            payload = None
        with self._lock:
            capture = self._captures[capture_id]
            self._bytes -= pcm.nbytes
            self._pending_samples[capture.speaker] -= len(pcm)
            if payload is not None and self._bytes + len(payload) <= self._budget:
                asset_id = hashlib.sha256(capture_id.encode() + str(start).encode() + payload).hexdigest()
                self._clips[asset_id] = Clip(asset_id, capture_id, capture.speaker, start, len(pcm), capture.anchor, payload)
                self._bytes += len(payload)
            else:
                self._record_loss(capture_id, start, start + len(pcm), "encode_failed" if payload is None else "buffer_full")

    def _end_speech(self, capture_id, end):
        if region := self._speech.pop(capture_id, None):
            self._regions.setdefault(capture_id, []).append((region[0], min(end, region[1] + 2400), region[2]))

    def _process(self, item):
        _, capture_id, start, samples = item
        capture = self._captures[capture_id]
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        blocks = [(start + offset, len(block), float(np.sqrt(np.mean(block ** 2))))
                  for offset in range(0, len(samples), 320) for block in [samples[offset:offset + 320]]]
        old_id = self._worker_capture.get(capture.speaker)
        if old_id and old_id != capture_id:
            self._encode(old_id, force=True)
            with self._lock:
                self._end_speech(old_id, self._captures[old_id].end)
        state = self._buffered.get(capture_id)
        if state and state[1] + len(state[0]) != start:
            self._encode(capture_id, force=True)
            state = None
        with self._lock:
            self._worker_capture[capture.speaker] = capture_id
            self._bytes -= samples.nbytes
            self._bytes += pcm.nbytes
            # Clip preroll and VAD on either side of a real gap, even if a
            # checkpoint already encoded the preceding PCM buffer.
            prev_end = self._worker_end.get(capture_id, capture.start)
            if prev_end != start:
                self._end_speech(capture_id, prev_end)
                self._voice_floor[capture_id] = start
            self._worker_end[capture_id] = start + len(samples)
            floor = self._voice_floor.get(capture_id, capture.start)
            for pos, count, rms in blocks:
                region = self._speech.get(capture_id)
                if rms >= QUIET_RMS:
                    clear = rms >= VOICE_RMS or bool(region and region[2])
                    self._speech[capture_id] = (region[0] if region else max(floor, pos - 2400), pos + count, clear)
                elif region and pos - region[1] >= 7200:
                    self._end_speech(capture_id, pos)
        self._buffered[capture_id] = (np.concatenate((state[0], pcm)), state[1]) if state else (pcm, start)
        self._encode(capture_id)

    def _run(self):
        while True:
            self._complete_checkpoints()
            if self._shutdown and self._encoded == self._accepted:
                return
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._process(item)
            except Exception:
                _, capture_id, start, samples = item
                _log.exception("Archive input failed capture=%s sample=%s", capture_id, start)
                with self._lock:
                    self._record_loss(capture_id, start, start + len(samples), "encode_failed")
            finally:
                self._processed = item[0]
                self._queue.task_done()

    def _complete_checkpoints(self):
        with self._lock:
            if not self._checkpoints or self._checkpoints[0] > self._processed:
                return
        for capture_id in list(self._buffered):
            self._encode(capture_id, force=True)
        with self._progress:
            self._encoded = self._processed
            while self._checkpoints and self._checkpoints[0] <= self._encoded:
                self._checkpoints.popleft()
            self._progress.notify_all()

    def checkpoint(self):
        """Fence already accepted PCM. Later arrivals cannot postpone this fence."""
        with self._lock:
            target = self._accepted
            if target > self._encoded and (not self._checkpoints or self._checkpoints[-1] < target):
                self._checkpoints.append(target)
            return target

    def encoded(self, checkpoint):
        with self._lock:
            return self._encoded >= checkpoint

    def flush(self, timeout=2):
        target = self.checkpoint()
        with self._progress:
            return self._progress.wait_for(lambda: self._encoded >= target, timeout=timeout)

    def seal(self):
        with self._lock:
            self._sealed = True
            return self.checkpoint()

    @staticmethod
    def _capture_at(captures, speaker, at):
        tracks = [c for c in captures if c.speaker == speaker and c.clock(c.start) <= at + 0.25]
        # Allow a late provider event just after the end of its track.
        tracks = [c for c in tracks if c.clock(c.end) >= at - 1.0]
        return max(tracks, key=lambda c: (c.clock(c.start) <= at, c.clock(c.start)), default=None)

    def _voice_ranges(self, capture_id):
        ranges = list(self._regions.get(capture_id, ()))
        if speech := self._speech.get(capture_id):
            ranges.append((speech[0], min(self._worker_end[capture_id], speech[1] + 2400), speech[2]))
        return ranges

    def speech_start(self, speaker, observed_start, capture_id=None, *, after=None, final_only=False):
        """Use the capture clock to seed cloud turn boundaries before the final arrives."""
        with self._lock:
            capture = (self._captures.get(capture_id) if capture_id else
                       self._capture_at(self._captures.values(), speaker, observed_start))
            if capture is None and final_only and not capture_id:
                recent_tracks = [c for c in self._captures.values() if c.speaker == speaker
                                 and c.clock(c.start) <= observed_start
                                 and c.clock(c.end) >= observed_start - LATE_FINAL_SECONDS]
                if len(recent_tracks) == 1:
                    capture = recent_tracks[0]
            if capture is None:
                return observed_start, capture_id
            observed = capture.sample(observed_start)
            floor = capture.sample(after) + 1 if after is not None else capture.start
            if not final_only:
                floor = max(floor, observed - RATE)
            recent = RATE * (LATE_FINAL_SECONDS if final_only else 1)
            matches = [left for left, right, _ in self._voice_ranges(capture.id)
                       if floor <= left <= observed + RATE // 4 and right >= observed - recent
                       and (left >= observed - RATE or right <= observed)]
            return (capture.clock(matches[-1]) if matches else observed_start), capture.id

    def first_speech(self, timed_out=()):
        with self._lock:
            starts = [capture.clock(left) for capture in self._captures.values()
                      for left, _, clear in self._voice_ranges(capture.id) if clear]
            starts.extend(capture.clock(capture.start) for capture in self._captures.values()
                          if capture.speaker in timed_out)
            return min(starts, default=None)

    def media(self, utterances, *, start, end, previous, previous_status, timed_out=(), previous_candidate=None):
        """Build a per-role playlist from sample ranges already through the encoder."""
        with self._lock:
            captures = dict(self._captures)
            clips = dict(self._clips)
            speech = {key: self._voice_ranges(key) for key in captures}
            losses = {key: tuple(ranges) for key, ranges in self._losses.items()}
        media, assets, states = {}, {}, {}
        for speaker in SPEAKERS:
            turns = [u for u in utterances if u["speaker"] == speaker]
            selected, approximate, reason = _select_audio_ranges(
                speaker, turns, captures, speech, losses, start, end, speaker in timed_out, previous_candidate,
            )
            retained = [item for item in previous.get(speaker, ()) if item["asset_id"] not in clips]
            manifest = retained.copy()
            for capture_id, ranges in selected.items():
                track_clips = sorted((c for c in clips.values() if c.capture_id == capture_id), key=lambda c: c.sample_start)
                saved_ranges = [(item["sample_start"], item["sample_start"] + item["sample_count"])
                                for item in retained if item["capture_id"] == capture_id]
                for left, right in _merge_ranges(ranges):
                    covered = sum(max(0, min(right, b) - max(left, a)) for a, b in saved_ranges)
                    for clip in track_clips:
                        begin = max(left, clip.sample_start)
                        finish = min(right, clip.sample_start + clip.sample_count)
                        if begin >= finish:
                            continue
                        manifest.append(dict(asset_id=clip.asset_id, offset=(begin - clip.sample_start) / RATE,
                                             duration=(finish - begin) / RATE, sample_start=begin,
                                             sample_count=finish - begin, capture_id=capture_id,
                                             capture_mono=clip.anchor + begin / RATE))
                        assets[clip.asset_id] = clip.payload
                        covered += finish - begin
                    if covered < right - left:
                        reason = reason or "samples_missing"

            if retained and previous_status.get(speaker, {}).get("state") == "partial":
                reason = reason or previous_status[speaker].get("reason", "samples_missing")
            approximate |= bool(retained and previous_status.get(speaker, {}).get("state") == "approximate")
            duration_ms = round(sum(item["sample_count"] for item in manifest) * 1000 / RATE)
            state = "partial" if reason and manifest else "missing" if reason else (
                "approximate" if approximate and manifest else "ready" if manifest else "none"
            )
            media[speaker] = sorted(manifest, key=lambda item: item["capture_mono"])
            states[speaker] = {"state": state, "duration_ms": duration_ms}
            if reason:
                states[speaker]["reason"] = reason
        return media, assets, states

    def release(self, asset_ids):
        with self._lock:
            for asset_id in asset_ids:
                clip = self._clips.pop(asset_id, None)
                if clip:
                    self._bytes -= len(clip.payload)

    def prune(self, before_mono):
        with self._lock:
            expired = [c.asset_id for c in self._clips.values() if c.started_mono + c.sample_count / RATE < before_mono]
            self.release(expired)
            # Keep the matching hints while the Opus bytes themselves can be
            # freed after the round was committed.
            retain_hints_since = before_mono - LATE_FINAL_SECONDS
            for key, capture in list(self._captures.items()):
                self._regions[key] = [r for r in self._regions.get(key, ()) if capture.clock(r[1]) >= retain_hints_since]
                if (key not in self._active.values() and key not in self._worker_capture.values()
                        and self._capture_ticket.get(key, 0) <= self._encoded
                        and capture.clock(capture.end) < retain_hints_since):
                    del self._captures[key]
                    for history in (self._regions, self._losses, self._speech, self._worker_end,
                                    self._voice_floor, self._capture_ticket):
                        history.pop(key, None)

    def close(self, timeout=3):
        self.seal()
        self._shutdown = True
        self._thread.join(timeout)
        return not self._thread.is_alive()
