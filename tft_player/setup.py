"""Game settings the player relies on, applied before each match.

TFT is an Unreal client; its display settings live in GameUserSettings.ini and
are read at launch (and rewritten on exit), so we set them while the game is
not running:
- borderless fullscreen at the screen's own resolution (the HUD positions scale from 1080p)
- 100% render resolution: at 50% text and icons are blurred upscales
A backup of the user's original file is kept next to it.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

INI = Path(os.environ.get('LOCALAPPDATA', '')) / 'TFT' / 'Saved' / 'Config' / 'WindowsClient' / 'GameUserSettings.ini'

WANT = {
    # borderless fullscreen at the panel's native resolution: sharp (no 1080p
    # upscaling blur), not capped like windowed mode at 180% scaling, and
    # overlays (MetaTFT via Overwolf) can draw on it - exclusive fullscreen blocks them
    'FullscreenMode': '1', 'LastConfirmedFullscreenMode': '1',     # 1 = windowed fullscreen (borderless)
    'sg.ResolutionQuality': '100',
}


def native_resolution():
    """The primary screen's current resolution in physical pixels (Windows display scaling does not change it), or
    None. It was a constant 2560x1600 - the development PC's screen - which another player's screen would not have."""
    try:
        import win32api
        mode = win32api.EnumDisplaySettings(None, -1)           # ENUM_CURRENT_SETTINGS
        return int(mode.PelsWidth), int(mode.PelsHeight)
    except Exception:
        return None


def ensure_settings(path: Path = INI):
    """Returns a list of the keys changed (empty if already right)."""
    if not path.exists():
        return []
    want = dict(WANT)
    res = native_resolution()
    if res:
        want.update({'ResolutionSizeX': str(res[0]), 'ResolutionSizeY': str(res[1]),
                     'LastUserConfirmedResolutionSizeX': str(res[0]), 'LastUserConfirmedResolutionSizeY': str(res[1])})
    text = path.read_text(encoding='utf-8')
    changed = []
    for key, val in want.items():
        pat = re.compile(rf'^{re.escape(key)}=.*$', re.M)
        if pat.search(text):
            new = pat.sub(f'{key}={val}', text)
            if new != text:
                changed.append(key)
                text = new
    if changed:
        backup = path.with_suffix('.ini.user-backup')
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text, encoding='utf-8')
    return changed
