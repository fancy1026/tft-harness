"""TFT window, continuous capture and native input.

Safety kept from the old controller, minus the per-step overhead:
- only acts while the TFT client is the foreground window
- F8 latches a stop (checked before every input); held buttons are released
Everything uses 1080p reference coordinates; mapping to the real client
happens here.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

TFT_EXES = {'tftclient-win64-shipping.exe', 'tftclient.exe'}
VK_F8 = 0x77
VK_F7 = 0x76

user32 = ctypes.WinDLL('user32', use_last_error=True)
try:
    user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    pass

import pydirectinput  # noqa: E402  (after DPI awareness)
import win32gui  # noqa: E402
import win32process  # noqa: E402
import psutil  # noqa: E402

pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False


class Stopped(RuntimeError):
    pass


class Paused(RuntimeError):
    """An input tried while F7 has the bot paused: nothing is sent."""


# F7 pauses and continues: while set, no input is sent (Input.check) and no model question goes out
# (coach.ask_model); the player keeps reading the game and tracking its board, and on continue asks the model
# again with the state as it is then (Player.paused_tick)
PAUSED = threading.Event()


STOP_HOOKS = []          # called once after F8 has killed the children, before the hard exit


def _kill_switch():
    """F8 ends the whole process, whatever it is doing: queueing, waiting for a window or
    the model, mid-drag, in a background thread. Polled every 15 ms from import on, so a
    tap is never missed (checked only before an input, F8 did nothing while the player
    waited - 2026-09-29). Buttons are released, child processes killed, then a hard exit."""
    import os
    import sys
    user32.GetAsyncKeyState(VK_F8)          # clear the 'pressed since last call' bit
    f7_down = False
    while True:
        down = bool(user32.GetAsyncKeyState(VK_F7) & 0x8000)
        if down and not f7_down:
            if PAUSED.is_set():
                PAUSED.clear()
                msg = '\nF7: continuing\n'
            else:
                PAUSED.set()
                try:
                    for b in ('left', 'right'):
                        pydirectinput.mouseUp(button=b)
                except Exception:
                    pass
                msg = '\nF7: paused - F7 to continue, F8 to stop\n'
            try:
                sys.stdout.write(msg)
                sys.stdout.flush()
            except Exception:
                pass
        f7_down = down
        if user32.GetAsyncKeyState(VK_F8) & 0x8001:
            try:
                for b in ('left', 'right'):
                    pydirectinput.mouseUp(button=b)
            except Exception:
                pass
            try:
                for c in psutil.Process().children(recursive=True):
                    c.kill()
            except Exception:
                pass
            try:
                sys.stdout.write('\nF8: stopped\n')
                sys.stdout.flush()
            except Exception:
                pass
            for hook in STOP_HOOKS:                # e.g. reopen the model's conversation for questions
                try:
                    t = _threading.Thread(target=hook, daemon=True)
                    t.start()
                    t.join(5)
                except Exception:
                    pass
            os._exit(3)
        time.sleep(0.015)


import os as _os  # noqa: E402
import threading as _threading  # noqa: E402

_threading.Thread(target=_kill_switch, daemon=True, name='f8-kill-switch').start()


def bring_to_front(hwnd) -> bool:
    """Put the TFT window in front (test runs only: the machine is dedicated to TFT)."""
    import win32con
    if win32gui.GetForegroundWindow() == hwnd:
        return True
    kernel32 = ctypes.WinDLL('kernel32')
    fg = win32gui.GetForegroundWindow()
    fg_thread = win32process.GetWindowThreadProcessId(fg)[0] if fg else 0
    me = kernel32.GetCurrentThreadId()
    try:
        if fg_thread and fg_thread != me:
            user32.AttachThreadInput(fg_thread, me, True)
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.BringWindowToTop(hwnd)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            # Windows may refuse; a bare Alt tap makes this process eligible
            user32.keybd_event(0x12, 0, 0, 0)
            user32.keybd_event(0x12, 0, 2, 0)
            win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    finally:
        if fg_thread and fg_thread != me:
            user32.AttachThreadInput(fg_thread, me, False)
    time.sleep(0.15)
    if win32gui.GetForegroundWindow() != hwnd and fg and _shell_flyout(fg):
        # Quick settings / Start / Search keep the foreground against SetForegroundWindow: the
        # whole of game 20261010-093812 was played by nobody behind Quick settings. Escape closes them.
        user32.keybd_event(0x1B, 0, 0, 0)
        user32.keybd_event(0x1B, 0, 2, 0)
        time.sleep(0.3)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        time.sleep(0.15)
    return win32gui.GetForegroundWindow() == hwnd


def _shell_flyout(h) -> bool:
    """A Windows shell surface (Quick settings, notification centre, Start, Search) in front."""
    try:
        cls = win32gui.GetClassName(h)
    except Exception:
        return False
    return cls in ('ControlCenterWindow', 'Windows.UI.Core.CoreWindow', 'Shell_TrayWnd',
                   'NotifyIconOverflowWindow', 'XamlExplorerHostIslandWindow')


@dataclass
class Client:
    hwnd: int
    left: int
    top: int
    width: int
    height: int

    def to_screen(self, p):
        """1080p reference point -> screen. HUD scales with height; most of it is
        centred, the item bench is left-anchored ('L'), the player list right ('R')."""
        s = self.height / 1080
        anchor = p[2] if len(p) > 2 else 'C'
        if anchor == 'L':
            x = p[0] * s
        elif anchor == 'R':
            x = self.width - (1920 - p[0]) * s
        else:
            x = self.width / 2 + (p[0] - 960) * s
        return self.left + round(x), self.top + round(p[1] * s)

    def from_screen(self, pt, anchor='C'):
        """Screen point -> 1080p reference point in the given anchor."""
        s = self.height / 1080
        x, y = (pt[0] - self.left) / s, (pt[1] - self.top) / s
        if anchor == 'R':
            x = 1920 - (self.width / s - x)
        elif anchor != 'L':
            x = 960 + x - self.width / s / 2
        return x, y


def live_rect(hwnd):
    """The window's client area on screen now: (left, top, width, height), or None."""
    try:
        _, _, w, h = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (0, 0))
        return x, y, w, h
    except Exception:
        return None


def settled_client(timeout=120.0, steady=3.0):
    """The TFT window once its size has held for `steady` s: it opens as a small window and only then goes
    borderless fullscreen (run 20261010-002419: measured at 1729x1123 while opening, captured that crop for the
    whole game and never read a stage - 25 minutes blind, 8th)."""
    end, last, since = time.time() + timeout, None, None
    while time.time() < end:
        c = find_client()
        rect = (c.left, c.top, c.width, c.height) if c else None
        if rect and rect == last:
            if time.time() - since >= steady:
                return c
        else:
            last, since = rect, time.time()
        time.sleep(0.5)
    return find_client()


def find_client() -> Client | None:
    found = []

    def visit(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            exe = psutil.Process(pid).name().lower()
        except Exception:
            return
        if exe in TFT_EXES:
            _, _, w, h = win32gui.GetClientRect(hwnd)
            if w >= 640 and h >= 360:
                x, y = win32gui.ClientToScreen(hwnd, (0, 0))
                found.append(Client(hwnd, x, y, w, h))

    win32gui.EnumWindows(visit, None)
    return max(found, key=lambda c: c.width * c.height) if found else None


def _fast_capture_recovery():
    """When Windows takes the screen image away (a scene change, a display-mode switch), dxcam retries
    with a growing wait: 0.25 s steps for 5 s, then 2 s steps - the screen came back to us seconds after
    the game showed it again. Retry every 0.05 s instead, and print dxcam's recovery messages so the run
    log shows each loss."""
    import logging
    from dxcam.core import dxgi_errors
    dxgi_errors.ProgressiveWait.next_delay_seconds = lambda self: 0.05
    lg = logging.getLogger('dxcam')
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('CAPTURE: %(message)s'))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)


class Capture:
    """Always-running DXGI capture of the client region; read() returns the
    newest frame (RGB numpy) and its capture time.

    One pump thread takes every frame from dxcam and keeps the newest; every reader gets each new frame. dxcam's
    get_latest_frame clears one shared "new frame" flag, so with several readers (the game loop, the Watcher, the
    carousel recorder) each frame went to only one of them and the loop waited for the next (the user, 2026-10-11:
    carousel looks 0.6-1.1 s apart; replayed on 32 recorded carousels the chaser got its target 26 times at a look
    every 0.35 s and 0 times at 1.0 s)."""

    STALL = 0.5                     # a read that waited this long for a frame is logged

    def __init__(self, client: Client, fps=30):
        import dxcam
        _fast_capture_recovery()
        self.client = client
        self.stalls = []                # (time, seconds waited): the loop logs and empties it
        self.cam = dxcam.create(output_color='RGB')
        # exclusive fullscreen switches the display mode, but Windows may still
        # report the old window size: trust the capture output's real size
        ow, oh = self.cam.width, self.cam.height
        if client.width > ow or client.height > oh or (client.left <= 0 and client.top <= 0
                                                        and client.width >= ow):
            client.left, client.top, client.width, client.height = 0, 0, ow, oh
        region = (client.left, client.top, client.left + client.width, client.top + client.height)
        self.fps = fps
        self.cam.start(region=region, target_fps=fps, video_mode=True)
        self._last = None
        self.raw = None
        self._rect_at = time.time()
        import threading
        self._cond = threading.Condition()
        self._latest = None             # (raw, perception frame, capture time, sequence number)
        self._seq = 0
        self._seen = threading.local()  # per reader: the last sequence number it was given
        self._win_lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._pump, name='capture-pump', daemon=True).start()

    def _pump(self):
        """Every new frame once: downscaled once for perception (at most 1080 high), kept as the newest."""
        import cv2
        while self._running:
            try:
                frame = self.cam.get_latest_frame()
            except Exception:
                frame = None
            if frame is None:
                time.sleep(0.02)          # stopped (a window change restarts it) or not started yet
                continue
            raw = np.asarray(frame)
            h, w = raw.shape[:2]
            small = cv2.resize(raw, (round(w * 1080 / h), 1080), interpolation=cv2.INTER_AREA) if h > 1100 else raw
            with self._cond:
                self._seq += 1
                self._latest = (raw, small, time.time(), self._seq)
                self._cond.notify_all()

    def follow_window(self):
        """The window moved or changed size (it went fullscreen after we measured it): capture its new area,
        and update the client in place so input mapping follows. Checked every 2 s by read()."""
        if not self._win_lock.acquire(blocking=False):
            return False                  # another reader is checking it right now
        try:
            return self._follow_window()
        finally:
            self._win_lock.release()

    def _follow_window(self):
        self._rect_at = time.time()
        rect = live_rect(self.client.hwnd)
        if not rect or rect[2] < 640 or rect[3] < 360:
            return False
        c = self.client
        ow, oh = self.cam.width, self.cam.height
        x, y, w, h = rect
        if w > ow or h > oh or (x <= 0 and y <= 0 and w >= ow):
            x, y, w, h = 0, 0, ow, oh
        if (x, y, w, h) == (c.left, c.top, c.width, c.height):
            return False
        print(f'WINDOW: the TFT window is now {w}x{h} at {x},{y} (was {c.width}x{c.height} at {c.left},{c.top}) - '
              f'capturing the new area', flush=True)
        try:
            self.cam.stop()
        except Exception:
            pass
        c.left, c.top, c.width, c.height = x, y, w, h
        self.cam.start(region=(x, y, x + w, y + h), target_fps=self.fps, video_mode=True)
        self._last = None
        return True

    def read(self):
        if time.time() - getattr(self, '_rect_at', 0) > 2.0:
            self.follow_window()
        return self._read()

    def _read(self):
        """Newest frame for perception, at most 1080 high (the calibration
        height; a native 2560x1600 frame becomes 1728x1080). The untouched
        frame is kept in .raw for full-resolution snapshots. Input mapping uses
        the real window size, so the downscale never affects clicks."""
        t0 = time.time()
        seen = getattr(self._seen, 'seq', 0)
        with self._cond:                  # a frame this reader has not had yet (each reader gets every frame)
            self._cond.wait_for(lambda: self._latest is not None and self._latest[3] != seen, timeout=1.0)
            latest = self._latest
        waited = time.time() - t0
        if waited > self.STALL:
            self.stalls.append((round(t0, 2), round(waited, 2)))
            print(f'CAPTURE: no frame for {waited:.1f}s', flush=True)
        if latest is not None and latest[3] != seen:
            self._seen.seq = latest[3]
            self._seen.raw = latest[0]
            self.raw = latest[0]
            self._last = (latest[1], latest[2])
        return self._last

    def read_raw(self):
        """The newest full-resolution frame this reader has not had yet (the carousel recorder), or None."""
        got = self._read()
        return getattr(self._seen, 'raw', None) if got is not None else None

    def close(self):
        self._running = False
        try:
            self.cam.stop()
        except Exception:
            pass


def window_desc(hwnd) -> str:
    """'title' [class] process.exe - to name whatever took the foreground or sits under a click."""
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        import psutil
        exe = psutil.Process(pid).name()
    except Exception:
        exe = '?'
    try:
        return f"{win32gui.GetWindowText(hwnd)!r} [{win32gui.GetClassName(hwnd)}] {exe}"
    except Exception:
        return f'hwnd {hwnd} {exe}'


def overwolf_running() -> bool:
    """Overwolf (the MetaTFT companion) injects an overlay into the game that can take its
    mouse and keyboard input (run20: ~30 of 45 inputs lost from 2-5 to 3-5 with it running)."""
    try:
        import psutil
        return any((p.info['name'] or '').lower().startswith('overwolf') for p in psutil.process_iter(['name']))
    except Exception:
        return False


user32.GetKeyboardLayout.restype = ctypes.c_void_p
user32.LoadKeyboardLayoutW.restype = ctypes.c_void_p
WM_INPUTLANGCHANGEREQUEST = 0x0050


def game_layout(hwnd) -> int:
    """The keyboard layout (HKL) of the game window's thread: low word 0x0804 = Chinese, where an
    IME would turn the F/D/E/W hotkeys into composition text."""
    tid, _ = win32process.GetWindowThreadProcessId(hwnd)
    return user32.GetKeyboardLayout(tid) or 0


class Input:
    def __init__(self, client: Client, dry_run=False, auto_focus=False):
        self.client = client
        self.dry_run = dry_run
        self.auto_focus = auto_focus
        self.stopped = False
        self.focus_steals = []           # times another window had the foreground when we sent input
        self.blocked = []                # (time, point, window) a click target covered by another window
        self.layout_fixes = []           # (time, layout) the game's keyboard layout was not Latin
        self.log = []

    # -- guards ----------------------------------------------------------------
    def check(self):
        if self.stopped or user32.GetAsyncKeyState(VK_F8) & 0x8000:
            self.stopped = True
            raise Stopped('F8 stop')
        if PAUSED.is_set():
            raise Paused('F7 pause')
        if not self.dry_run:
            self.release_stuck_buttons()
        if not self.dry_run and win32gui.GetForegroundWindow() != self.client.hwnd:
            other = window_desc(win32gui.GetForegroundWindow())
            if not (self.auto_focus and bring_to_front(self.client.hwnd)):
                from tft_player.player import NotForeground
                raise NotForeground()
            # the game drops input that arrives just as it regains focus: record it and let it settle
            self.focus_steals.append((time.time(), other))
            print(f'INPUT: focus taken back from {other}')
            time.sleep(0.3)


    def release_stuck_buttons(self):
        """No mouse button may be down before an input of ours (we never hold one between
        inputs). A held right button made the game ignore every click and drag for 470 s in
        run22; a release from here could not clear it, only a real click did. Release what we
        can; if it stays down, say so and don't send the input."""
        held = [b for b, vk in (('left', 1), ('right', 2)) if user32.GetAsyncKeyState(vk) & 0x8000]
        if not held:
            return
        for b in held:
            pydirectinput.mouseUp(button=b)
        time.sleep(0.05)
        still = [b for b, vk in (('left', 1), ('right', 2)) if user32.GetAsyncKeyState(vk) & 0x8000]
        if still:
            if time.time() - getattr(self, '_held_said', 0) > 5:
                self._held_said = time.time()
                print(f'INPUT: the {"/".join(still)} mouse button is held down outside the bot (a physical '
                      f'button, or Mouse Without Borders from another PC) - the game ignores clicks until it '
                      f'is released: click it once')
            from tft_player.player import NotForeground
            raise NotForeground()

    def keep_latin_layout(self):
        """Hotkeys need a non-IME layout in the game window: a Chinese one gets switched to US."""
        if self.dry_run:
            return
        hkl = game_layout(self.client.hwnd)
        if hkl & 0xFFFF in (0x0804, 0x0404, 0x0411, 0x0412):     # zh-CN, zh-TW, ja, ko: IME layouts
            us = user32.LoadKeyboardLayoutW('00000409', 0)
            win32gui.PostMessage(self.client.hwnd, WM_INPUTLANGCHANGEREQUEST, 0, us)
            self.layout_fixes.append((time.time(), hex(hkl)))
            print(f'INPUT: the game window had keyboard layout {hex(hkl)} (IME): switched to US')
            time.sleep(0.2)

    def diagnose(self, p=None) -> dict:
        """The input conditions right now, for a failed action: foreground, the game's keyboard
        layout, where the cursor is and which window is under it (and under the target)."""
        out = {}
        try:
            out['foreground'] = window_desc(win32gui.GetForegroundWindow())
            out['game_foreground'] = win32gui.GetForegroundWindow() == self.client.hwnd
            out['layout'] = hex(game_layout(self.client.hwnd))
            cx, cy = win32gui.GetCursorPos()
            out['cursor'] = (cx, cy)
            out['under_cursor'] = window_desc(user32.GetAncestor(win32gui.WindowFromPoint((cx, cy)), 2))
            if p is not None:
                tx, ty = self.client.to_screen(p)
                out['target'] = (tx, ty)
                out['under_target'] = window_desc(user32.GetAncestor(win32gui.WindowFromPoint((tx, ty)), 2))
            out['buttons_down'] = [b for b, vk in (('left', 1), ('right', 2)) if user32.GetAsyncKeyState(vk) & 0x8000]
        except Exception as e:
            out['error'] = repr(e)
        return out

    def clear_at(self, p):
        """The window under the click point must be the game's: another top window there takes
        the click. Name it, put the game on top, and check again."""
        if self.dry_run:
            return
        x, y = self.client.to_screen(p)
        for attempt in range(2):
            h = win32gui.WindowFromPoint((x, y))
            root = user32.GetAncestor(h, 2) if h else 0          # GA_ROOT
            if not root or root == self.client.hwnd:
                return
            desc = window_desc(root)
            self.blocked.append((time.time(), (x, y), desc))
            print(f'INPUT: click point {x},{y} is covered by {desc} - bringing the game to the top')
            bring_to_front(self.client.hwnd)
            win32gui.BringWindowToTop(self.client.hwnd)
            time.sleep(0.2)

    def ensure_front(self):
        """Called every tick: a covered TFT window also means unreadable frames."""
        if self.auto_focus and not self.dry_run and win32gui.GetForegroundWindow() != self.client.hwnd:
            other = window_desc(win32gui.GetForegroundWindow())
            ok = bring_to_front(self.client.hwnd)
            self.focus_steals.append((time.time(), other))
            print(f'INPUT: the game lost the foreground to {other}; taken back: {ok}')
            return ok
        return True

    def _record(self, kind, *args):
        self.log.append((time.time(), kind, args))

    # -- primitives (1080p reference coordinates) ---------------------------
    def move(self, p):
        x, y = self.client.to_screen(p)
        if not self.dry_run:
            user32.SetCursorPos(x, y)
            # plus a real mouse-move event (a 1px SendInput nudge there and back): after a
            # bare SetCursorPos the game can keep its old hover target, and E then sold a
            # different unit than the one under the cursor (action test, 2026-09-27).
            # Absolute SendInput moves are ignored on this machine.
            pydirectinput.moveRel(1, 0, relative=True)
            pydirectinput.moveRel(-1, 0, relative=True)
            for attempt in range(4):
                cx, cy = win32gui.GetCursorPos()
                if abs(cx - x) <= 3 and abs(cy - y) <= 3:
                    break
                # something moved it away (Mouse Without Borders parks it at the screen edge)
                print(f'INPUT: cursor is at {cx},{cy}, not at the target {x},{y} - placing it again')
                time.sleep(0.05)
                user32.SetCursorPos(x, y)
            else:
                from tft_player.player import NotForeground
                raise NotForeground()          # the click would land elsewhere: don't send it

    def hover(self, p, steps=8, duration=0.15):
        """Glide the cursor to p: hover tooltips need motion over the target."""
        self.check()
        self._record('hover', p)
        if self.dry_run:
            return
        pt = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        x0, y0 = self.client.from_screen((pt.x, pt.y), p[2] if len(p) > 2 else 'C')
        for i in range(1, steps + 1):
            self.move((x0 + (p[0] - x0) * i / steps, y0 + (p[1] - y0) * i / steps) + tuple(p[2:]))
            time.sleep(duration / steps)

    def click(self, p, button='left'):
        self.check()
        self.clear_at(p)
        self._record('click', p, button)
        if self.dry_run:
            return
        self.move(p)
        time.sleep(0.02)
        pydirectinput.mouseDown(button=button)
        time.sleep(0.03)
        pydirectinput.mouseUp(button=button)

    KEY_HOLD = 0.06

    def key(self, k, at=None):
        """Press a game hotkey; `at` hovers a point first (E sell, W place)."""
        self.check()
        self.keep_latin_layout()
        self._record('key', k, at)
        if self.dry_run:
            return
        if at is not None:
            self.move(at)
            time.sleep(0.03)
        pydirectinput.keyDown(k)
        # held over two game frames: on this machine under load (~15-25 fps) a 30 ms tap went unanswered - E sells
        # failed on 12-30% of sells from 2026-10-13 on, 0-7% before, most with time left and the unit hovered
        time.sleep(self.KEY_HOLD)
        pydirectinput.keyUp(k)

    def drag(self, a, b, duration=0.12):
        self.check()
        self.clear_at(a)
        self.clear_at(b)
        self._record('drag', a, b)
        if self.dry_run:
            return
        self.move(a)
        time.sleep(0.03)
        pydirectinput.mouseDown()
        try:
            time.sleep(0.04)
            steps = 6
            for i in range(1, steps + 1):
                self.move((a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps))
                time.sleep(duration / steps)
            time.sleep(0.03)
        finally:
            pydirectinput.mouseUp()

    def drag_hover(self, a, b, hover=1.0):
        """Press on a, move onto b, stay there `hover` seconds, release: how the game links two things (the user,
        2026-10-10: the BFF to its Sprykin rider, Taric to his ally - a progress bar fills while hovering, ~1 s)."""
        self.check()
        self.clear_at(a)
        self.clear_at(b)
        self._record('drag_hover', a, b)
        if self.dry_run:
            return
        self.move(a)
        time.sleep(0.03)
        pydirectinput.mouseDown()
        try:
            time.sleep(0.06)
            for i in range(1, 7):
                self.move((a[0] + (b[0] - a[0]) * i / 6, a[1] + (b[1] - a[1]) * i / 6))
                time.sleep(0.02)
            time.sleep(hover)
        finally:
            pydirectinput.mouseUp()


class ScreenRecorder:
    """A full video of the game (the user, 2026-10-10: every game recorded, a real video, not frames): ffmpeg in
    its own process grabs the display by Desktop Duplication and encodes on the GPU (Intel Quick Sync), so the
    player's loop pays little. 15 fps (30 took capture from ~25 to 15 fps in run 20261010-235723, with the
    profiler also on), scaled to 1080 high. Fragmented mp4: the file plays even if the process
    is killed mid-game. ffmpeg comes from the imageio-ffmpeg package."""

    def __init__(self, path, output_idx=0, fps=15):
        import os
        import subprocess
        import imageio_ffmpeg
        self.path = path
        hq = os.environ.get('TFT_VIDEO_HQ') == '1'      # --video-hq: 30 fps, native size, quality 20
        fps = 30 if hq else fps
        size = 'format=nv12' if hq else 'w=-1:h=1080:format=nv12'
        quality = '20' if hq else '28'
        self.log = open(str(path) + '.ffmpeg.log', 'w', encoding='utf-8')
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'warning', '-y',
             '-init_hw_device', 'd3d11va', '-filter_complex',
             f'ddagrab=output_idx={output_idx}:framerate={fps},hwmap=derive_device=qsv,'
             f'vpp_qsv={size}',
             '-c:v', 'h264_qsv', '-global_quality', quality, '-movflags', '+frag_keyframe+empty_moov', str(path)],
            stdin=subprocess.PIPE, stdout=self.log, stderr=self.log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def stop(self):
        """'q' lets ffmpeg finish the file; killed if it does not within 10 s."""
        if self.proc.poll() is None:
            try:
                self.proc.communicate(b'q', timeout=10)
            except Exception:
                self.proc.kill()
        self.log.close()


_PROFILER = []


def start_profiler(path, rate=50):
    """A sampling profile of this whole process (every thread, waiting included) by py-spy in its own process,
    for finding what slows the loop (the user, 2026-10-10: slow reads, 1.5-2x the earlier games). It reads our
    memory from outside without pausing us, and writes `path` (speedscope JSON, tools/profile_report.py) when
    this process exits. One per process: a multi-game run profiles all its games into the first run folder."""
    import os
    import subprocess
    if _PROFILER:
        return None
    exe = os.path.join(sys.prefix, 'Scripts', 'py-spy.exe')
    if not os.path.exists(exe):
        raise FileNotFoundError(exe)
    log = open(str(path) + '.log', 'w', encoding='utf-8')
    _PROFILER.append(subprocess.Popen(
        [exe, 'record', '--pid', str(os.getpid()), '--rate', str(rate), '--threads', '--idle', '--nonblocking',
         '--format', 'speedscope', '-o', str(path)],
        stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
    return path
