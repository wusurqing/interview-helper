"""Route modified Windows wheel events to the click-through prompt window.

The main process owns the window and sends line-delimited state over stdin.
Only scalar values are sent through the private Windows message.
"""
import argparse
import ctypes
import json
import sys
import threading

WM_MOUSEWHEEL = 0x020A
WM_MOUSEHWHEEL = 0x020E
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_LWIN = 0x5B
VK_RWIN = 0x5C
MODIFIERS = {"alt": VK_MENU, "ctrl": VK_CONTROL, "shift": VK_SHIFT}
MODIFIER_KEYS = (VK_MENU, VK_CONTROL, VK_SHIFT, VK_LWIN, VK_RWIN)


def parse_state(value):
    disabled = {"active": False, "scrollable": False, "generation": 0}
    if not isinstance(value, dict):
        return disabled
    try:
        hwnd = int(value.get("hwnd", "0"))
        message = value.get("message")
        generation = value.get("generation")
        modifier = value.get("modifier")
        if (
            type(value.get("active")) is not bool
            or type(value.get("scrollable")) is not bool
            or type(message) is not int
            or not 0x8000 <= message <= 0xBFFF
            or type(generation) is not int
            or not 1 <= generation <= 0x7FFFFFFF
            or not 0 < hwnd < 2 ** (ctypes.sizeof(ctypes.c_void_p) * 8)
            or modifier not in MODIFIERS
        ):
            return disabled
    except (TypeError, ValueError, OverflowError):
        return disabled
    return {
        "active": value["active"], "scrollable": value["scrollable"],
        "hwnd": hwnd, "message": message, "generation": generation,
        "modifier": modifier,
    }


def route_wheel(*, state, message, x, y, delta, pressed, bounds, post):
    if message != WM_MOUSEWHEEL or not state.get("active") or not state.get("scrollable") or not delta:
        return False
    required_key = MODIFIERS[state["modifier"]]
    if not pressed(required_key) or any(pressed(key) for key in MODIFIER_KEYS if key != required_key):
        return False
    rect = bounds(state["hwnd"])
    if not rect or not (rect[0] <= x < rect[2] and rect[1] <= y < rect[3]):
        return False
    return bool(post(state["hwnd"], state["message"], state["generation"], delta))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    if sys.platform != "win32" or args.parent_pid <= 0:
        return 2
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    result_t = ctypes.c_ssize_t
    hook_proc = ctypes.WINFUNCTYPE(result_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

    class MouseEvent(ctypes.Structure):
        _fields_ = [
            ("pt", wintypes.POINT), ("mouseData", wintypes.DWORD),
            ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
            ("extra", ctypes.c_size_t),
        ]

    user32.SetWindowsHookExW.argtypes = (ctypes.c_int, hook_proc, wintypes.HINSTANCE, wintypes.DWORD)
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    user32.CallNextHookEx.restype = result_t
    user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
    user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.IsWindow.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    user32.PostMessageW.restype = wintypes.BOOL
    user32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    user32.GetMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
    user32.GetMessageW.restype = ctypes.c_int
    user32.PeekMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT)
    dwmapi.DwmGetWindowAttribute.argtypes = (wintypes.HWND, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD)
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    parent = kernel32.OpenProcess(0x00100000, False, args.parent_pid)
    if not parent:
        print(json.dumps({"type": "error", "message": "cannot open parent process"}), flush=True)
        return 1

    state = {"active": False, "scrollable": False, "generation": 0}
    thread_id = kernel32.GetCurrentThreadId()
    msg = wintypes.MSG()
    # Ensure this thread's Windows queue exists before the reader can stop it.
    user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)

    def bounds(hwnd):
        if kernel32.WaitForSingleObject(parent, 0) != 0x102:
            return None
        if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
            return None
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != args.parent_pid:
            return None
        rect = wintypes.RECT()
        if dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
            return None
        return rect.left, rect.top, rect.right, rect.bottom

    def pressed(key):
        return user32.GetAsyncKeyState(key) < 0

    @hook_proc
    def on_mouse(code, message, data):
        if code >= 0 and message == WM_MOUSEWHEEL:
            try:
                event = ctypes.cast(data, ctypes.POINTER(MouseEvent)).contents
                delta = ctypes.c_short((event.mouseData >> 16) & 0xFFFF).value
                if route_wheel(
                    state=state, message=message, x=event.pt.x, y=event.pt.y,
                    delta=delta, pressed=pressed, bounds=bounds, post=user32.PostMessageW,
                ):
                    return 1
            except Exception:
                pass
        return user32.CallNextHookEx(None, code, message, data)

    hook = user32.SetWindowsHookExW(14, on_mouse, None, 0)
    if not hook:
        print(json.dumps({"type": "error", "message": "cannot install mouse hook"}), flush=True)
        kernel32.CloseHandle(parent)
        return 1

    def read_controls():
        nonlocal state
        try:
            for line in sys.stdin:
                try:
                    command = json.loads(line)
                except (ValueError, TypeError):
                    state = {"active": False, "scrollable": False, "generation": 0}
                    continue
                if isinstance(command, dict) and command.get("type") == "stop":
                    break
                if isinstance(command, dict) and command.get("type") == "state":
                    state = parse_state(command)
                    print(json.dumps({"type": "state", "generation": state["generation"]}), flush=True)
        finally:
            state = {"active": False, "scrollable": False, "generation": 0}
            user32.PostThreadMessageW(thread_id, 0x0012, 0, 0)

    print(json.dumps({"type": "ready"}), flush=True)
    threading.Thread(target=read_controls, daemon=True).start()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    finally:
        user32.UnhookWindowsHookEx(hook)
        kernel32.CloseHandle(parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
