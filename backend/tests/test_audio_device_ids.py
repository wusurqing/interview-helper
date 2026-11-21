from types import SimpleNamespace

import pytest

from services import audio


@pytest.fixture
def enumerate_devices(monkeypatch):
    speakers = []
    monkeypatch.setattr(audio, "_HAS_SC", True)
    monkeypatch.setattr(audio, "_sc", SimpleNamespace(
        all_speakers=lambda: speakers,
        default_speaker=lambda: speakers[0],
    ), raising=False)
    monkeypatch.setattr(audio, "_SC_DEVICE_CACHE", {})
    monkeypatch.setattr(audio, "_SC_ID_MAP", {})

    def enumerate_snapshot(*devices):
        speakers[:] = [SimpleNamespace(id=guid, name=name, channels=2) for guid, name in devices]
        entries = audio._get_sc_loopback_devices()
        return {audio._lookup_sc_speaker_guid(entry["id"]): entry["id"] for entry in entries}

    return enumerate_snapshot


def test_new_first_speaker_does_not_shift_existing_device_ids(enumerate_devices):
    before = enumerate_devices(("a", "Monitor"), ("b", "Headphones"))
    assert before == {"a": 20000, "b": 20001}
    after = enumerate_devices(("c", "Dock"), ("a", "Monitor"), ("b", "Headphones"))
    assert after == {**before, "c": 20002}


def test_absent_device_id_cannot_select_another_speaker(enumerate_devices):
    first = enumerate_devices(("a", "Monitor"), ("b", "Headphones"))
    assert enumerate_devices() == {}
    assert audio._lookup_sc_speaker_guid(first["a"]) is None
    assert audio._lookup_sc_speaker_guid(first["b"]) is None

    second = enumerate_devices(("c", "Dock"), ("b", "Headphones"))
    assert second["b"] == first["b"]
    assert second["c"] not in first.values()
    assert audio._lookup_sc_speaker_guid(first["a"]) is None
    assert enumerate_devices(("a", "Monitor"))["a"] == first["a"]


def test_name_fallback_only_replaces_unambiguous_missing_guid(enumerate_devices):
    first = enumerate_devices(("a", "Monitor"), ("b", "Headphones"))
    second = enumerate_devices(("c", "Monitor"), ("a", "Renamed monitor"), ("d", "Headphones"))
    assert second["a"] == first["a"]
    assert second["d"] == first["b"]
    assert second["c"] not in first.values()

    same_name = enumerate_devices(("e", "Headphones"), ("d", "Headphones"))
    assert same_name["d"] == second["d"]
    assert same_name["e"] not in second.values()
    replacement = enumerate_devices(("f", "Headphones"))
    assert replacement["f"] not in same_name.values()


def test_enumeration_failure_preserves_last_successful_snapshot(enumerate_devices, monkeypatch):
    first = enumerate_devices(("a", "Monitor"))

    def unavailable():
        raise RuntimeError("temporary device query failure")

    monkeypatch.setattr(audio._sc, "all_speakers", unavailable)
    assert audio._get_sc_loopback_devices() == []
    assert audio._lookup_sc_speaker_guid(first["a"]) == "a"
