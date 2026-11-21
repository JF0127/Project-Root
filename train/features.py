import json

import numpy as np
import torch

from .data import CACHE, CATALOG, ROLES, ExpressionDataset, _allowed_file, _role_pairs

STRICT_TIER = 'tier_1_strict'
EXPRESSION = {f'E{i:02d}': i for i in range(8)}
INTENSITY = {'low': 0, 'medium': 1, 'high': 1}
AUDIO_FIELDS = ('hidden_mean', 'hidden_std', 'hidden_tail')
INPUT_DIMS = {'text': 384, 'audio': 2304, 'video': 162}
STD_FLOOR = 1e-6


def _labels(split, groups):
    ids, tiers, expressions, intensities, eligibility = [], [], [], [], []
    for tid, pair in sorted(groups.items()):
        tier, = {row['tier'] for row in pair.values()}
        expr, intensity = [-1, -1], [-1, -1]
        for i, role in enumerate(ROLES):
            row = pair[role]
            if split == 'train' and row['high_conf_train_eligible'] is not True:
                continue
            expr[i] = EXPRESSION[row['expression_id']]
            intensity[i] = INTENSITY[row['intensity']]
        if max(expr) >= 0:
            ids.append(tid)
            tiers.append(tier)
            expressions.append(expr)
            intensities.append(intensity)
            eligibility.append([pair[role]['high_conf_train_eligible'] is True for role in ROLES])
    return ExpressionDataset(np.array(ids), np.array(tiers), np.array(expressions, dtype=np.int64),
                             np.array(intensities, dtype=np.int64), {}, np.array(eligibility))


def _aligned_indices(cache, groups, selected):
    ids = cache['turn_id'].tolist()
    where = {tid: i for i, tid in enumerate(ids)}
    if len(where) != len(ids) or (groups is not None and where.keys() != groups.keys()):
        raise ValueError('Cache turn_id must uniquely match the complete split annotation table')
    return np.array([where[tid] for tid in selected], dtype=np.int64)


def _video_vectors(cache, indices):
    neutral = cache['neutral_index'].item()
    if not 0 <= neutral < 52 or cache['blendshape_names'][neutral] != '_neutral':
        raise ValueError('Invalid neutral_index in video cache')
    blend = np.delete(cache['frame_blendshapes'], neutral, axis=1)
    pose, valid = cache['frame_head_pose'], cache['valid_mask']
    vectors = np.zeros((len(indices), 162), dtype=np.float64)
    for i, (start, stop) in enumerate(cache['turn_offsets'][indices]):
        mask = valid[start:stop]
        angles = pose[start:stop][mask].astype(np.float64)
        if not len(angles):
            continue
        angles = np.rad2deg(np.unwrap(np.deg2rad(angles), axis=0))
        frames = np.column_stack((blend[start:stop][mask], angles))
        vectors[i, :54] = frames.mean(axis=0)
        half = len(frames) // 2
        if half:
            vectors[i, 54:108] = frames.std(axis=0)
            vectors[i, 108:] = frames[half:].mean(axis=0) - frames[:half].mean(axis=0)
    return vectors


def cached_features(split, turn_ids, modalities, groups=None):
    if split not in ('train', 'val'):
        raise ValueError('Only Train and Val caches are available')
    features = {}
    for name in modalities:
        path = _allowed_file(f'{CACHE}/{split}/{name}_features.npz')
        with np.load(path, allow_pickle=False) as cache:
            order = _aligned_indices(cache, groups, turn_ids)
            if name == 'video':
                values = _video_vectors(cache, order)
            elif name == 'audio':
                values = np.stack([cache[key][order] for key in AUDIO_FIELDS], axis=2)
            else:
                values = cache['embedding'][order]
        features[name] = torch.from_numpy(values)
    return features


def _load_split(split, groups):
    data = _labels(split, groups)
    data.features = cached_features(split, data.turn_ids, INPUT_DIMS, groups)
    return data


def prepare(features, normalization, device):
    if 'video' in features:
        video = features['video']
        features['video'] = (video - video.new_tensor(normalization['mean'])) / video.new_tensor(normalization['std'])
    return {k: v.to(device=device, dtype=torch.float32) for k, v in features.items()}


def load_validation(normalization, device):
    catalog = json.loads(_allowed_file(CATALOG).read_text(encoding='utf-8'))
    val = _load_split('val', _role_pairs(catalog, 'val'))
    val.features = prepare(val.features, normalization, device)
    return val


def val_subsets(val):
    return {'val_strict': np.flatnonzero(val.tiers == STRICT_TIER), 'val_all': np.arange(len(val.turn_ids))}


def load_data(device):
    catalog = json.loads(_allowed_file(CATALOG).read_text(encoding='utf-8'))
    train_rows, val_rows = (_role_pairs(catalog, split) for split in ('train', 'val'))
    if train_rows.keys() & val_rows.keys():
        raise ValueError('Train and Val have overlapping turn_id')
    train, val = _load_split('train', train_rows), _load_split('val', val_rows)
    if np.count_nonzero(train.expr >= 0) != 1941:
        raise ValueError('Expected exactly 1941 eligible Train role records')
    if (val.expr.shape != (192, 2) or (val.expr < 0).any()
            or np.count_nonzero(val.tiers == STRICT_TIER) != 99):
        raise ValueError('Expected 192 complete Val turns including 99 strict turns')
    video = train.features['video']
    mean, std = video.mean(dim=0), video.std(dim=0, correction=0).clamp_min(STD_FLOOR)
    normalization = {'mean': mean.tolist(), 'std': std.tolist(), 'std_floor': STD_FLOOR}
    for data in (train, val):
        data.features = prepare(data.features, normalization, device)
    return train, val, normalization
