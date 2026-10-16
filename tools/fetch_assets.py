"""The game's own files the harness needs, downloaded at install instead of shipped in the repository:

- CommunityDragon's TFT game data         -> data/sources/cdragon-tft-<date>.json
- the factual catalogue built from it     -> data/factual-catalogue-18.json
- champion portraits, overlay item icons  -> data/champ-icons, data/item-icons-overlay (tools/fetch_champ_icons.py)
- official item icons                     -> data/item-assets/*.png (the list: data/item-assets/catalogue.json)

    python tools/fetch_assets.py           (install.bat runs it; files already there are kept)
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
GAME_DATA = 'https://raw.communitydragon.org/latest/cdragon/tft/en_us.json'
UA = {'User-Agent': 'Mozilla/5.0'}


def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def game_data():
    """The newest game data on disk, or today's download."""
    sources = ROOT / 'data' / 'sources'
    sources.mkdir(parents=True, exist_ok=True)
    have = sorted(sources.glob('cdragon-tft-*.json'))
    if have:
        return have[-1]
    path = sources / f'cdragon-tft-{time.strftime("%Y-%m-%d")}.json'
    raw = get(GAME_DATA, timeout=180)
    json.loads(raw)                                    # a broken download fails here, not in the player
    path.write_bytes(raw)
    print(f'game data: {path.relative_to(ROOT)} ({len(raw) // 1024} KB)')
    return path


def item_icons():
    """The official item icons the recogniser starts from, as data/item-assets/catalogue.json lists them."""
    folder = ROOT / 'data' / 'item-assets'
    items = json.loads((folder / 'catalogue.json').read_text(encoding='utf-8'))['items']
    got = failed = 0
    for it in items:
        path = folder / Path(it['path'].replace('\\', '/')).name
        if path.exists():
            continue
        try:
            path.write_bytes(get(it['url']))
            got += 1
        except Exception as e:
            failed += 1
            print(f"{it['name']}: {e}")
    print(f'item icons: {got} new, {len(list(folder.glob("*.png")))} in all' + (f', {failed} failed' if failed else ''))


def main():
    src = game_data()
    if not (ROOT / 'data' / 'factual-catalogue-18.json').exists():
        import build_factual_catalogue
        build_factual_catalogue.main(src)
    item_icons()                 # before the overlay icons: the item list fetch_champ_icons asks for loads these
    import fetch_champ_icons
    fetch_champ_icons.main()


if __name__ == '__main__':
    main()
