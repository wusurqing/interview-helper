"""Compatibility for SoundCard's Windows property-store binding."""


def configure_property_buffers(binding) -> None:
    """Use the full COM variant size when SoundCard declares only the pointer member."""
    if not hasattr(binding, "_PropVariant"):
        return
    ffi = binding._ffi
    if not {"PROPVARIANT", "BLOB_PROPVARIANT"}.issubset(ffi.list_types()[0]):
        return
    if ffi.sizeof("PROPVARIANT") >= ffi.sizeof("BLOB_PROPVARIANT"):
        return

    # SoundCard 0.4.6 uses this wrapper for both the endpoint name and format.
    # COM may write the entire union even when returning only a string.
    ole32 = binding._ole32

    def release(buffer):
        try:
            ole32.PropVariantClear(ffi.cast("PROPVARIANT *", buffer))
        finally:
            ole32.CoTaskMemFree(buffer)

    allocate = ffi.new_allocator(ole32.CoTaskMemAlloc, release, should_clear_after_alloc=True)

    class PropertyVariant:
        def __init__(self):
            self._storage = allocate("BLOB_PROPVARIANT *")
            self.ptr = ffi.cast("PROPVARIANT *", self._storage)

    binding._PropVariant = PropertyVariant
