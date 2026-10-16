"""Extract factual set data from a CommunityDragon source; no strategy or network calls.

    python tools/build_factual_catalogue.py [data/sources/cdragon-tft-<date>.json]   (default: the frozen 09-18 one)
The release builds it at install from the source it downloads (tools/fetch_assets.py)."""
import hashlib
import json
import sys
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'data/sources/cdragon-tft-2026-09-18.json'


def plain(value):
    return re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',value or '').replace('&nbsp;',' ')).strip()


def main(source=None):
    source=Path(source) if source else FROZEN
    raw=json.loads(source.read_text(encoding='utf-8'))
    roster=json.loads((ROOT/'data/units-18.json').read_text(encoding='utf-8'))['units']
    selected=raw['sets']['18']
    champions={}
    for name,cost in roster.items():
        normalize=lambda v:re.sub(r'[^a-z0-9]','',v.lower())
        named=[c for c in selected['champions'] if normalize(c['name'])==normalize(name)]
        # by name and cost; a patch may change a cost, so a single champion of that name also counts
        matches=[c for c in named if c['cost']==cost] or (named if len(named)==1 else [])
        if len(matches)!=1: raise ValueError(f'Ambiguous or absent champion: {name}')
        c=matches[0]
        champions[name]=dict(source_name=c['name'],base_cost=c['cost'],traits=c['traits'],role=c.get('role'),stats=c['stats'],
            ability=dict(name=c['ability']['name'],description=plain(c['ability']['desc'])))
    names={t for c in champions.values() for t in c['traits']}
    traits={t['name']:dict(description=plain(t['desc']),thresholds=[e['minUnits'] for e in t['effects']])
            for t in selected['traits'] if t['name'] in names}
    augments={}
    for a in raw['items']:
        if not a.get('isAugment') or not a.get('apiName','').startswith(('TFT18_', 'TFT_Augment_')): continue
        augments.setdefault(a['name'],[]).append(dict(api_name=a['apiName'],description=plain(a.get('desc'))))
    date=re.search(r'\d{4}-\d{2}-\d{2}',source.name)
    result=dict(set=18,source_file=source.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_date=date.group(0) if date else None,live_patch_verified=False,
        scope='Factual frozen catalogue. Missing roles and unresolved formula tokens remain explicit. No preferred rosters or item holders.',
        champions=champions,traits=traits,augments=augments)
    path=ROOT/'data/factual-catalogue-18.json'; path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(champions=len(champions),traits=len(traits),augments=len(augments),path=str(path))))


if __name__=='__main__': main(sys.argv[1] if len(sys.argv)>1 else None)
