import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("prompt_mouse_hook", Path(__file__).with_name("prompt_mouse_hook.py"))
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class PromptWheelRoutingTests(unittest.TestCase):
    def setUp(self):
        self.state = hook.parse_state({
            "type": "state", "active": True, "scrollable": True, "modifier": "alt",
            "hwnd": "12345", "message": 32799, "generation": 7,
        })
        self.keys = {hook.VK_MENU}
        self.posted = []

    def route(self, **overrides):
        args = {
            "state": self.state, "message": hook.WM_MOUSEWHEEL, "x": -350, "y": 75,
            "delta": -30, "pressed": lambda key: key in self.keys,
            "bounds": lambda hwnd: (-400, 0, -100, 200),
            "post": lambda *args: self.posted.append(args) or True,
        }
        args.update(overrides)
        return hook.route_wheel(**args)

    def test_matching_partial_delta_posts_to_overlay_and_is_consumed(self):
        self.assertTrue(self.route())
        self.assertEqual(self.posted, [(12345, 32799, 7, -30)])

    def test_every_other_event_passes_through(self):
        self.assertFalse(self.route(message=hook.WM_MOUSEHWHEEL))
        self.assertFalse(self.route(message=0x201))
        self.assertFalse(self.route(x=-100))
        self.assertFalse(self.route(y=200))
        self.assertFalse(self.route(bounds=lambda hwnd: None))
        self.keys.add(hook.VK_SHIFT)
        self.assertFalse(self.route())
        self.keys = {hook.VK_CONTROL}
        self.assertFalse(self.route())
        self.assertEqual(self.posted, [])

    def test_capture_requires_active_scrollable_and_successful_delivery(self):
        self.assertFalse(self.route(state={**self.state, "scrollable": False}))
        self.assertFalse(self.route(state={**self.state, "active": False}))
        self.assertFalse(self.route(post=lambda *args: False))

    def test_invalid_control_input_fails_closed_for_capture(self):
        for patch in (
            {"modifier": "bogus"}, {"message": 1}, {"generation": -1},
            {"hwnd": "wrong"}, {"active": 1}, {"scrollable": "yes"},
        ):
            with self.subTest(patch=patch):
                state = hook.parse_state({**self.state, "type": "state", **patch})
                self.assertFalse(state["active"])

    def test_ctrl_shift_and_windows_keys_count_as_extra_modifiers(self):
        for key in (hook.VK_CONTROL, hook.VK_SHIFT, hook.VK_LWIN, hook.VK_RWIN):
            with self.subTest(key=key):
                self.keys = {hook.VK_MENU, key}
                self.assertFalse(self.route())


if __name__ == "__main__":
    unittest.main()
