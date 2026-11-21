import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'
METADATA = 'dataset/metadata'
CACHE = 'dataset/features'
CATALOG = f'{METADATA}/dataset_catalog.json'
MANIFEST = f'{METADATA}/phase1_model_inputs_manifest.jsonl'
SPLITS = {'train': ('training', 1000), 'val': ('validation', 192)}
ROLES = ('R01', 'R02')
COLUMNS = ('turn_id', 'split', 'tier', 'speaker_id', 'session_id', 'role_id',
           'expression_id', 'intensity', 'high_conf_train_eligible')


def model_inputs(batch):
    return {key: batch[key] for key in ('text', 'audio', 'video', 'role') if key in batch}


@dataclass
class ExpressionDataset(Dataset):
    turn_ids: np.ndarray
    tiers: np.ndarray
    expr: np.ndarray
    intensity: np.ndarray
    features: dict[str, torch.Tensor]
    eligible: np.ndarray

    @cached_property
    def records(self):
        return np.argwhere(self.expr >= 0)

    @cached_property
    def by_turn(self):
        groups = [[] for _ in self.turn_ids]
        for index, (turn, _) in enumerate(self.records):
            groups[turn].append(index)
        return groups

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        turn, role = self.records[index]
        return {**{key: values[turn] for key, values in self.features.items()}, 'role': role,
                'expression': self.expr[turn, role], 'intensity': self.intensity[turn, role],
                'label_valid': self.expr[turn, role] >= 0, 'intensity_valid': self.expr[turn, role] > 0,
                'high_conf_train_eligible': self.eligible[turn, role], 'turn_id': str(self.turn_ids[turn])}

    def loader(self, turns, batch_turns):
        batches = ([r for turn in turns[i:i + batch_turns] for r in self.by_turn[turn]]
                   for i in range(0, len(turns), batch_turns))
        return DataLoader(self, batch_sampler=list(batches), generator=torch.Generator())


@dataclass(frozen=True)
class Turn:
    turn_id: str
    split: str
    tier: str
    speaker_id: str
    session_id: str
    expressions: tuple[str, str]
    intensities: tuple[str, str]
    dual_high_conf: bool
    video: Path
    audio: Path
    text: str
    audio_duration: float
    audio_sample_rate: int


def _allowed_file(relative):
    path = ROOT / relative
    if not path.is_relative_to(ROOT) or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError(f'Not a literal project file: {relative}')
    return path


def _role_pairs(catalog, split):
    usage, count = SPLITS[split]
    path = catalog['usage_rules_and_limitations'][usage]['path']
    if path != f'{METADATA}/phase2_production/{split}_{count}_annotations.parquet':
        raise ValueError(f'Unexpected {split} annotation path: {path}')
    groups = defaultdict(dict)
    for row in pq.read_table(_allowed_file(path), columns=list(COLUMNS)).to_pylist():
        tid, role = row['turn_id'], row['role_id']
        if (row['split'] != split or not re.fullmatch(r'T_\d+', tid)
                or role not in ROLES or role in groups[tid]):
            raise ValueError(f'Invalid {split} annotation key: {tid}, {role}')
        groups[tid][role] = row
    if len(groups) != count or any(len(pair) != len(ROLES) for pair in groups.values()):
        raise ValueError(f'Expected {count} complete role pairs in {split}')
    return groups


def _speaker_file(relative, folder, filename):
    if relative != (folder / filename).as_posix():
        raise ValueError(f'Unexpected current-speaker input: {relative}')
    return _allowed_file(relative)


def _make_turn(tid, pair, inputs, folder):
    a, b = (pair[role] for role in ROLES)
    if any(a[k] != b[k] for k in ('tier', 'speaker_id', 'session_id')):
        raise ValueError(f'Role rows disagree on shared metadata: {tid}')
    audio = _speaker_file(inputs['speaker_audio'], folder, 'speaker_A.wav')
    video = _speaker_file(inputs['speaker_video'], folder, 'speaker_A.mp4')
    text = inputs['transcript']
    if not isinstance(text, str):
        raise ValueError(f'Non-string transcript: {tid}')
    info = sf.info(audio)
    return Turn(turn_id=tid, split=a['split'], tier=a['tier'], speaker_id=a['speaker_id'],
                session_id=a['session_id'], expressions=(a['expression_id'], b['expression_id']),
                intensities=(a['intensity'], b['intensity']),
                dual_high_conf=all(r['high_conf_train_eligible'] is True for r in (a, b)),
                audio=audio, video=video, text=text, audio_duration=info.duration,
                audio_sample_rate=info.samplerate)


def load_turns(*splits: str) -> list[Turn]:
    if not splits or not set(splits) <= SPLITS.keys() or len(set(splits)) != len(splits):
        raise ValueError('Explicitly request train, val, or both without repetitions')
    catalog = json.loads(_allowed_file(CATALOG).read_text(encoding='utf-8'))
    media = Path(catalog['media_assets']['path'])
    if media != Path('dataset/turns'):
        raise ValueError(f'Unexpected speaker media root: {media}')
    manifest, = (r['path'] for r in catalog['frozen_files_inventory'] if r['path'] == MANIFEST)
    groups = {}
    for split in splits:
        pairs = _role_pairs(catalog, split)
        if groups.keys() & pairs.keys():
            raise ValueError('Requested splits contain overlapping turn IDs')
        groups.update(pairs)
    turns = {}
    with _allowed_file(manifest).open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            tid = row['turn_id']
            if tid not in groups:
                continue
            if tid in turns:
                raise ValueError(f'Duplicate input manifest turn_id: {tid}')
            turns[tid] = _make_turn(tid, groups[tid], row['model_inputs'], media / tid)
    if turns.keys() != groups.keys():
        raise ValueError(f'Missing model inputs for {sorted(groups.keys() - turns.keys())}')
    return [turns[tid] for tid in sorted(turns)]
