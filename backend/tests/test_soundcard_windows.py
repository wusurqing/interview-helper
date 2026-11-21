"""Check property-buffer ownership without opening a Windows audio device."""

from types import SimpleNamespace

from cffi import FFI

from services.soundcard_windows import configure_property_buffers


def make_binding(*, full_layout):
    ffi = FFI()
    ffi.cdef("typedef struct { unsigned short tag[4]; void *payload[2]; } BLOB_PROPVARIANT;")
    if full_layout:
        ffi.cdef("typedef BLOB_PROPVARIANT PROPVARIANT;")
    else:
        ffi.cdef("typedef struct { unsigned short tag[4]; void *payload; } PROPVARIANT;")
    return SimpleNamespace(_ffi=ffi, _PropVariant=object)


def test_full_property_layout_keeps_the_upstream_wrapper():
    binding = make_binding(full_layout=True)
    configure_property_buffers(binding)
    assert binding._PropVariant is object


def test_short_property_layout_uses_zeroed_storage_and_releases_contents_before_buffer():
    binding = make_binding(full_layout=False)
    ffi = binding._ffi
    allocated = []
    released = []

    def allocate(size):
        # A nonzero fill makes the allocator's initialization observable.
        storage = ffi.new("char[]", b"x" * size)
        allocated.append((size, storage))
        return storage

    def clear(ptr):
        released.append(("contents", ptr == allocated[0][1]))
        return 0

    def free(ptr):
        released.append(("buffer", ptr == allocated[0][1]))

    binding._ole32 = SimpleNamespace(CoTaskMemAlloc=allocate, PropVariantClear=clear, CoTaskMemFree=free)
    configure_property_buffers(binding)
    prop = binding._PropVariant()
    size = ffi.sizeof("BLOB_PROPVARIANT")
    assert allocated[0][0] == size
    assert ffi.buffer(prop.ptr, size)[:] == bytes(size)
    del prop
    assert released == [("contents", True), ("buffer", True)]
