"""Freeze scheme1_v1 from its original membership; existing outputs must match byte-for-byte."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'metadata'
DIAG = META / 'phase2_pilot' / 'diagnostics'
SPLIT = META / 'phase2_splits' / 'scheme1_v1'
STRICT, RELAXED = 'tier_1_strict', 'tier_2_relaxed'
SPLITS = ['train', 'val', 'test']


def require(ok, label):
    if not ok:
        raise ValueError(label)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def put(path, content, *, frozen=False):
    if path.exists() and frozen:
        require(path.read_bytes() == content, f'frozen artifact differs: {relative(path)}')
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('xb') as fp:
        fp.write(content)
    if frozen:
        tmp.chmod(0o444)
    tmp.replace(path)


def frozen_split(f):
    src = DIAG / 'taskB_scheme1_turns.csv'
    content = src.read_bytes()
    s = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False).set_index('turn_id', drop=False)
    require(len(s) == len(f) == 1358 and s.index.is_unique and set(s.index) == set(f.index),
                  'scheme 1 does not index exactly all 1358 turns')
    require(set(s['split']) == set(SPLITS), 'bad split names')
    fields = ['speaker_id', 'session_id', 'conversation_id', 'candidate_tier', 'prompt_id_unique', 'prompt_hash']
    require(f.loc[s.index, fields].eq(s[fields]).all().all(), 'scheme 1 provenance mismatch')
    require(s.roles.eq('R01;R02').all(), 'role list differs from fixed pair')
    roles = s[['split', 'roles']].assign(roles=s.roles.str.split(';')).explode('roles')
    require(len(roles) == 2716 and roles.groupby(level=0)['split'].nunique().eq(1).all() and
                  roles.groupby(level=0).roles.agg(set).map(lambda x: x == {'R01', 'R02'}).all(), 'pair invariant failed')
    require(s.speaker_id.nunique() == 19 and s.groupby('speaker_id')['split'].nunique().eq(1).all(), 'speaker leak')
    by = {k: s[s['split'].eq(k)] for k in SPLITS}
    se = {k: set(g.session_id) for k, g in by.items()}
    session_cross = {f'{x}_{y}': sorted(se[x] & se[y]) for i, x in enumerate(SPLITS) for y in SPLITS[i+1:]}
    require(not session_cross['train_val'] and not session_cross['train_test'] and
                  len(session_cross['val_test']) == 1, 'scheme 1 session invariant failed')
    holy = set(by['val'].speaker_id) | set(by['test'].speaker_id)
    expected = ['train' if r.split == 'train' else 'strict_eval' if r.candidate_tier == STRICT
                else 'relaxed_holdout' for r in s.itertuples()]
    require(s.eval_disposition.eq(expected).all() and not by['train'].speaker_id.isin(holy).any(),
                  'holdout leakage including relaxed tiers')
    cover = {k: dict(turns=len(g), roles=2*len(g), strict=int(g.candidate_tier.eq(STRICT).sum()),
                     relaxed=int(g.candidate_tier.eq(RELAXED).sum()),
                     speakers=sorted(set(g.speaker_id)), sessions=sorted(se[k]),
                     conversations=int(g.conversation_id.nunique()),
                     prompts=int(g.prompt_hash.nunique()), vendors=sorted(set(f.loc[g.index].vendor_id)))
             for k, g in by.items()}
    require([cover[k]['turns'] for k in SPLITS] == [1000, 192, 166], 'unexpected scheme 1 sizes')
    mapping = s[['prompt_id_unique', 'prompt_hash']].drop_duplicates()
    require(mapping.prompt_hash.is_unique and mapping.prompt_id_unique.is_unique and len(mapping) == 50,
                  'unexpected Prompt ID↔hash relationship')
    prompts = []
    for (pid, h), group in s.groupby(['prompt_id_unique', 'prompt_hash'], sort=True):
        count = Counter(group['split'])
        prompts.append(dict(prompt_id_unique=pid, prompt_hash=h, turns=len(group),
                            distinct_speakers=int(group.speaker_id.nunique()), distinct_sessions=int(group.session_id.nunique()),
                            **{k: count[k] for k in SPLITS},
                            cross_split=sum(bool(count[k]) for k in SPLITS) > 1))
    membership = SPLIT / 'turns.csv'
    lock = dict(version='scheme1_v1', source=relative(src), membership=relative(membership), sha256=sha(content),
                phase1_manifest_sha256=sha((META / 'phase1_model_inputs_manifest.jsonl').read_bytes()),
                atomic_keys=['turn_id', 'speaker_id'], role_pair_atomic=True, speaker_pairwise_overlap=0,
                covers=cover, session_overlaps=session_cross, prompt_cross_split_count=sum(r['cross_split'] for r in prompts),
                strict_heldout_eval_only=True, relaxed_val_test_protected={k: cover[k]['relaxed'] for k in ('val', 'test')},
                note='成员关系按原件字节冻结，不随 Expression 版本变化。成员冻结不代表评估无污染；见 v1.1 暴露审计。')
    put(membership, content, frozen=True)
    put(SPLIT / 'split_manifest.json', encode(lock), frozen=True)
    put(SPLIT / 'prompt_repeats.csv', pd.DataFrame(prompts).to_csv(index=False).encode(), frozen=True)
    return s, lock


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    source = META / 'phase1_model_inputs_manifest.jsonl'
    with source.open(encoding='utf8') as fh:
        rows = [json.loads(line) for line in fh]
    turns = pd.DataFrame([dict(turn_id=r['turn_id'], candidate_tier=r['candidate_tier'],
                               **r['source_provenance']) for r in rows]).set_index('turn_id')
    require(turns.index.is_unique, 'duplicate Phase 1 manifest key')
    _, manifest = frozen_split(turns)
    print(manifest['version'], manifest['sha256'])
