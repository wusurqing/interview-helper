"""Nonblocking capture subscription and bounded, memory-only Opus encoding."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import queue
import threading
import time
from uuid import uuid4

import numpy as np

RATE = 16000
BLOCK_SAMPLES = RATE * 2
ENCODE_AT = RATE * 8 // 5  # leave room for the next device callback under the 2 s cap


class CapturedAudio(np.ndarray):
    """An ndarray view which keeps source coordinates through queue slicing."""

    def __new__(cls, samples, capture_id, sample_start, anchor):
        value = np.asarray(samples).view(cls)
        value.capture_id = capture_id
        value.sample_start = sample_start
        value.capture_anchor = anchor
        return value

    def __array_finalize__(self, parent):
        self.capture_id = getattr(parent, "capture_id", "")
        self.sample_start = getattr(parent, "sample_start", 0)
        self.capture_anchor = getattr(parent, "capture_anchor", 0.0)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(value, CapturedAudio) and isinstance(item, slice):
            start, _, step = item.indices(len(self))
            if step == 1:
                value.sample_start = self.sample_start + start
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


class AudioRecorder:
    def __init__(self, warning, *, budget=64 * 1024 * 1024):
        self._warning = warning
        self._budget = budget
        self._queue = queue.Queue(maxsize=128)
        self._lock = threading.RLock()
        self._buffered = {}
        self._pending_samples = {"interviewer": 0, "candidate": 0}
        self._bytes = 0
        self._clips: dict[str, Clip] = {}
        self._regions: list[dict] = []
        self._speech = {}
        self._cloud_matches = {}
        self._fallback = {}
        self._closed = False
        self._flushing = threading.Event()
        self._thread = threading.Thread(target=self._run, name="archive.opus", daemon=True)
        self._thread.start()

    @property
    def memory_bytes(self):
        with self._lock:
            return self._bytes

    def offer(self, samples, speaker):
        if speaker not in self._pending_samples or not len(samples) or self._closed:
            return
        count = len(samples)
        # The callback does only a bounded copy/enqueue. No codec, file IO or VAD.
        with self._lock:
            if self._pending_samples[speaker] + count > BLOCK_SAMPLES or self._bytes + samples.nbytes > self._budget:
                self._warning("音频归档缓冲已满，部分录音未保存")
                return
            if getattr(samples, "capture_id", ""):
                capture, start, anchor = samples.capture_id, samples.sample_start, samples.capture_anchor
            else:
                capture, start, anchor = self._fallback.setdefault(speaker, (uuid4().hex, 0, time.monotonic() - count / RATE))
                self._fallback[speaker] = capture, start + count, anchor
            block = (speaker, capture, start, anchor, np.array(samples, copy=True, subok=False))
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                self._warning("音频编码队列已满，部分录音未保存")
                return
            self._pending_samples[speaker] += count
            self._bytes += block[-1].nbytes

    def _encode(self, key, force=False):
        state = self._buffered.get(key)
        if not state:
            return
        chunks, start, anchor = state
        count = len(chunks)
        if not force and count < ENCODE_AT:
            return
        pcm = chunks if force else chunks[:BLOCK_SAMPLES]
        try:
            payload = encode_opus(pcm)
        except Exception:
            with self._lock:
                self._bytes -= chunks.nbytes
                self._pending_samples[key[0]] -= len(chunks)
            self._buffered.pop(key, None)
            raise
        speaker, capture = key
        asset_id = hashlib.sha256(capture.encode() + str(start).encode() + payload).hexdigest()
        with self._lock:
            self._bytes -= pcm.nbytes
            self._pending_samples[speaker] -= len(pcm)
            if self._bytes + len(payload) <= self._budget:
                self._clips[asset_id] = Clip(asset_id, capture, speaker, start, len(pcm), anchor, payload)
                self._bytes += len(payload)
            else:
                self._warning("录音内存达到上限，部分录音未保存")
        if len(pcm) < count:
            self._buffered[key] = (chunks[len(pcm):].copy(), start + len(pcm), anchor)
        else:
            del self._buffered[key]

    def _process(self, item):
        speaker, capture, start, anchor, samples = item
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        key = speaker, capture
        blocks = [(start + offset, len(block), float(np.sqrt(np.mean(block ** 2))) >= 0.006)
                  for offset in range(0, len(samples), 320) if len(block := samples[offset:offset + 320])]
        with self._lock:
            self._bytes -= samples.nbytes
            self._bytes += pcm.nbytes
            # Lightweight sample-based VAD used only for archive alignment.
            for pos, count, voiced in blocks:
                region = self._speech.get(key)
                if voiced:
                    if region is None:
                        region = dict(speaker=speaker, capture_id=capture, start=max(0, pos - 2400), end=pos + count, anchor=anchor)
                        self._speech[key] = region
                    region["end"] = pos + count
                elif region and pos - region["end"] >= 7200:
                    self._regions.append({**region, "end": region["end"] + 2400})
                    del self._speech[key]
        state = self._buffered.get(key)
        if state and state[1] + len(state[0]) != start:
            self._encode(key, force=True)
            state = None
        self._buffered[key] = (np.concatenate((state[0], pcm)), state[1], state[2]) if state else (pcm, start, anchor)
        self._encode(key)

    def _run(self):
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                self._flush_if_requested()
                if self._closed:
                    return
                continue
            try:
                self._process(item)
            except Exception:
                self._warning("录音压缩失败，部分录音不可用")
            finally:
                self._queue.task_done()
            if self._queue.empty():
                self._flush_if_requested()

    def _flush_if_requested(self):
        if not self._flushing.is_set():
            return
        try:
            for key in list(self._buffered):
                self._encode(key, force=True)
        except Exception:
            self._warning("录音压缩失败")
        finally:
            self._flushing.clear()

    def flush(self, timeout=2):
        self._flushing.set()
        deadline = time.monotonic() + timeout
        while (self._queue.unfinished_tasks or self._flushing.is_set()) and time.monotonic() < deadline:
            time.sleep(0.005)
        return not self._queue.unfinished_tasks and not self._flushing.is_set()

    def media(self, utterances):
        """Build a playlist without reencoding. Trim overlaps by source samples."""
        result = {"interviewer": [], "candidate": []}
        assets = {}
        with self._lock:
            clips = list(self._clips.values())
        clips.sort(key=lambda c: c.started_mono)
        last_end = {}
        for utterance in sorted(utterances, key=lambda u: u["started_mono"]):
            speaker = utterance["speaker"]
            start_time = utterance["started_mono"]
            end_time = utterance.get("audio_ended_mono")
            if end_time is None:
                continue
            for clip in clips:
                if clip.speaker != speaker:
                    continue
                start = round((start_time - clip.anchor) * RATE)
                end = round((end_time - clip.anchor) * RATE)
                token = speaker, clip.capture_id
                left = max(start, clip.sample_start, last_end.get(token, 0))
                right = min(end, clip.sample_start + clip.sample_count)
                if right <= left:
                    continue
                result[speaker].append(dict(asset_id=clip.asset_id, offset=(left - clip.sample_start) / RATE,
                                            duration=(right - left) / RATE, sample_start=left,
                                            sample_count=right - left, capture_id=clip.capture_id,
                                            capture_mono=clip.anchor + left / RATE))
                assets[clip.asset_id] = clip.payload
                last_end[token] = right
        return result, assets

    def align(self, speaker, started_mono, key, *, final=False):
        """Match local speech intervals conservatively, not callback arrival time."""
        with self._lock:
            regions = self._regions + list(self._speech.values())
            owned = [r for r in regions if r.get("owner") == key]
            available = [r for r in regions if r["speaker"] == speaker and not r.get("owner")]
            start = self._cloud_matches.get(key, (started_mono, None))[0]
            eligible = [r for r in available if start - 8 <= r["anchor"] + r["start"] / RATE <= start + (3 if final else 0.25)]
            if not owned and not eligible:
                return self._cloud_matches.get(key, (started_mono, None))
            if not owned:
                owned = [min(eligible, key=lambda r: abs(r["anchor"] + r["start"] / RATE - start))]
            if final and eligible:
                tail = max(r["anchor"] + r["end"] / RATE for r in owned)
                for region in sorted(eligible, key=lambda r: r["anchor"] + r["start"] / RATE):
                    begin = region["anchor"] + region["start"] / RATE
                    if tail <= begin <= tail + 0.7:
                        owned.append(region)
                        tail = region["anchor"] + region["end"] / RATE
            if not owned:
                return started_mono, None
            for region in owned:
                region["owner"] = key
            aligned = (min(r["anchor"] + r["start"] / RATE for r in owned),
                       max(r["anchor"] + r["end"] / RATE for r in owned))
            self._cloud_matches[key] = aligned
            return aligned

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
            self._regions = [r for r in self._regions if r["anchor"] + r["end"] / RATE >= before_mono]
            self._cloud_matches = {k: v for k, v in self._cloud_matches.items() if v[1] >= before_mono}

    def close(self, timeout=3):
        self.flush(timeout)
        self._closed = True
        self._thread.join(timeout)
