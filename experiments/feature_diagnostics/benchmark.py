import csv
import os
from collections import Counter
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from time import perf_counter
from unicodedata import name

import av
import mediapipe as mp
import numpy as np
import sentence_transformers
import torch
import transformers

from train import encoders as enc
from train.data import MODELS, ROLES

SEED = 42
TIER_QUOTAS = {'tier_1_strict': 22, 'tier_2_relaxed': 6}
CLASSES = tuple(f'E{i:02}' for i in range(8))
DEVICES = ('cpu', 'cuda')
WARMUPS = 2
VIDEO_TIMES = ('decode_ms', 'rgb_roi_ms', 'fallback_rgb_ms', 'decode_roi_ms', 'mediapipe_init_ms',
               'srgb_wrap_ms', 'mediapipe_inference_ms', 'total_ms')
FACE_GROUPS = {
    'upper': [i for i, k in enumerate(enc.BLENDSHAPES) if k.startswith(('brow', 'eye', 'cheekSquint', 'noseSneer'))],
    'lower': [i for i, k in enumerate(enc.BLENDSHAPES) if k.startswith(('jaw', 'mouth', 'cheekPuff'))],
}


def _duration_band(turn):
    if turn.audio_duration < 1.5:
        return 'short'
    return 'long' if turn.audio_duration > 8 else 'regular'


def select_turns(train):
    if any(t.split != 'train' for t in train):
        raise ValueError('Probe-01 selection requires Train turns only')
    rng = np.random.default_rng(SEED)
    pool = [t for t in train if t.dual_high_conf and all(e in CLASSES for e in t.expressions)]
    labels = Counter((role, e) for t in pool for role, e in zip(ROLES, t.expressions, strict=True))
    tiers, bands, covered = Counter(), Counter(), Counter()
    speakers, sessions = set(), set()
    selected = []

    def score(turn):
        needs = [key for key in zip(ROLES, turn.expressions, strict=True) if covered[key] < 2]
        return (sum(5 + 20 / labels[key] for key in needs) + 3 * (bands[_duration_band(turn)] < 3)
                + 2 * (len(speakers) < 8 and turn.speaker_id not in speakers)
                + (len(sessions) < 3 and turn.session_id not in sessions) + rng.random())

    for _ in range(sum(TIER_QUOTAS.values())):
        turn = max((t for t in pool if tiers[t.tier] < TIER_QUOTAS[t.tier]), key=score)
        pool.remove(turn)
        selected.append(turn)
        covered.update(zip(ROLES, turn.expressions, strict=True))
        tiers[turn.tier] += 1
        bands[_duration_band(turn)] += 1
        speakers.add(turn.speaker_id)
        sessions.add(turn.session_id)
    if (min(covered[r, e] for r in ROLES for e in CLASSES) < 2 or len(speakers) < 8
            or len(sessions) < 2 or set(bands) != {'short', 'regular', 'long'} or tiers != TIER_QUOTAS):
        raise ValueError(f'Selection did not satisfy coverage: {covered}, {bands}, {tiers}')
    return sorted(selected, key=lambda t: t.turn_id)


def _summary(rows, key):
    values = [r[key] for r in rows]
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)))


@contextmanager
def _measure(times, stage, *, cuda=False):
    if cuda:
        torch.cuda.synchronize()
    start = perf_counter()
    yield
    if cuda:
        torch.cuda.synchronize()
    times[f'{stage}_ms'] += (perf_counter() - start) * 1000
    if cuda:
        times['peak_allocated_vram_bytes'] = torch.cuda.max_memory_allocated()


def _torch_profiles(model, turns, encode, *, audio=False):
    profiles, features = {}, []
    for device in DEVICES:
        model.to(device)
        cuda = device == 'cuda'
        for _ in range(WARMUPS):
            encode(turns[0], nullcontext)
        rows, peaks = [], []
        for turn in turns:
            if cuda and audio:
                torch.cuda.reset_peak_memory_stats()
            times = Counter()
            feature = encode(turn, partial(_measure, times, cuda=cuda))
            ms = times['inference_ms']
            row = dict(turn_id=turn.turn_id, inference_ms=ms)
            if audio:
                row['rtf'] = ms / 1000 / turn.audio_duration
            if cuda:
                features.append(feature)
                if audio:
                    peaks.append(times['peak_allocated_vram_bytes'])
            rows.append(row)
        profile = dict(inference_ms=_summary(rows, 'inference_ms'), per_turn=rows)
        if audio:
            profile['rtf'] = _summary(rows, 'rtf')
        if peaks:
            profile['peak_allocated_vram_bytes'] = max(peaks)
        profiles[device] = profile
    model.to('cpu')
    torch.cuda.empty_cache()
    return profiles, features


def _audio_benchmark(turns):
    encoder = enc.AudioEncoder(MODELS)
    for _ in range(WARMUPS):
        encoder.prepare(turns[0])
    prepared, rows = {}, []
    for turn in turns:
        times = Counter()
        prepared[turn.turn_id] = encoder.prepare(turn, partial(_measure, times))
        rows.append(dict(turn_id=turn.turn_id, **times, read_resample_ms=times.total()))

    def encode(turn, stage):
        return encoder.encode(turn, prepared=prepared[turn.turn_id], stage=stage)

    profiles, features = _torch_profiles(encoder.model, turns, encode, audio=True)
    return features, dict(model_id=enc.WAVLM_ID, runtime_config=encoder.config, profiles=profiles,
                          prepare_ms={k: _summary(rows, k) for k in ('read_ms', 'resample_ms', 'read_resample_ms')},
                          preparation_per_turn=rows, precision='float32',
                          timing_scope='model forward, input transfer and pooling excluded',
                          valid_turns=sum(r['is_valid'] for r in features),
                          short_turns=int(sum(r['is_short_turn'] for r in features)),
                          trimmed_turns=int(sum(r['speech_end_sec'] < r['duration_sec'] for r in features)))


def _scan_text(model, train):
    texts = [t.text for t in train]
    lengths = enc.token_lengths(model, texts)
    n = len(train)
    non_ascii = sum(not text.isascii() for text in texts)
    non_latin = sum(any(c.isalpha() and 'LATIN' not in name(c, '') for c in text) for text in texts)
    truncated = sum(k > model.max_seq_length for k in lengths)
    scan = dict(turns=n, empty=sum(not text.strip() for text in texts), non_ascii_count=non_ascii,
                non_ascii_fraction=non_ascii / n, suspected_non_english_count=non_latin,
                suspected_non_english_fraction=non_latin / n,
                language_heuristic='at least one non-Latin alphabetic character',
                token_mean=float(np.mean(lengths)), token_p95=float(np.percentile(lengths, 95)),
                token_max=max(lengths), truncated_count=truncated, truncated_fraction=truncated / n,
                max_seq_length=model.max_seq_length)
    return dict(zip((t.turn_id for t in train), lengths, strict=True)), scan


def _text_benchmark(train, turns):
    model = enc.load_minilm(MODELS)
    lengths, scan = _scan_text(model, train)

    def encode(turn, stage):
        return enc.encode_text(turn, model, token_len=lengths[turn.turn_id], stage=stage)

    profiles, features = _torch_profiles(model, turns, encode)
    return features, dict(model_id=enc.MINILM_ID, train_scan=scan, profiles=profiles, precision='float32',
                          timing_scope='encode including tokenization, normalization and return to host',
                          finite_turns=int(sum(np.isfinite(r['embedding']).all() for r in features)),
                          sample_empty=sum(r['is_empty'] for r in features),
                          sample_truncated=sum(r['is_truncated'] for r in features))


def _dynamics(sample):
    valid, t = sample['valid_mask'], sample['frame_timestamps']
    left = t < (t[0] + t[-1]) / 2
    both_halves = (valid & left).any() and (valid & ~left).any()
    result = {}
    for group, indexes in FACE_GROUPS.items():
        values = sample['frame_blendshapes'][:, indexes]
        delta = None
        if both_halves:
            delta = float(np.abs(values[valid & left].mean(axis=0) - values[valid & ~left].mean(axis=0)).mean())
        result[group] = dict(mean_std=float(values[valid].std(axis=0).mean()) if valid.sum() >= 2 else None,
                             delta=delta)
    return result


def _video_summary(rows):
    frames = sum(r['n_frames'] for r in rows)
    valid = sum(r['n_valid'] for r in rows)
    motion = {}
    for group in FACE_GROUPS:
        motion[group] = {}
        for metric in ('mean_std', 'delta'):
            values = [r['dynamics'][group][metric] for r in rows if r['dynamics'][group][metric] is not None]
            motion[group][metric] = dict(mean=float(np.mean(values)) if values else None, turns=len(values))
    return dict(timings_ms={k: _summary(rows, k) for k in VIDEO_TIMES}, frames=frames, valid_frames=valid,
                valid_fraction=valid / frames, valid_values_finite=all(r['valid_values_finite'] for r in rows),
                crop_modes=dict(Counter(r['crop_mode'] for r in rows)), dynamics=motion,
                warnings=[r['turn_id'] for r in rows if r['valid_fraction'] < 0.85], per_turn=rows)


def _video_benchmark(turns):
    if (len(enc.BLENDSHAPES), len(FACE_GROUPS['upper']), len(FACE_GROUPS['lower'])) != (52, 23, 28):
        raise ValueError('Wrong blendshape partition')
    options = enc.face_options(MODELS)
    profiles, features = {}, []
    for fps in (5, enc.CACHE_FPS):
        for _ in range(WARMUPS):
            enc.encode_video(turns[0], options, fps)
        rows = []
        for turn in turns:
            times = Counter()
            with _measure(times, 'total'):
                sample = enc.encode_video(turn, options, fps, stage=partial(_measure, times))
            times['decode_roi_ms'] = times['decode_ms'] + times['rgb_roi_ms']
            mask = sample['valid_mask']
            valid = int(mask.sum())
            finite = all(np.isfinite(sample[k][mask]).all() for k in ('frame_blendshapes', 'frame_head_pose'))
            rows.append(dict(turn_id=turn.turn_id, tier=turn.tier, speaker_id=turn.speaker_id,
                             crop_mode=sample['crop_mode'], n_frames=len(mask), n_valid=valid,
                             valid_fraction=valid / len(mask), valid_values_finite=finite,
                             dynamics=_dynamics(sample), **{k: times[k] for k in VIDEO_TIMES}))
            if fps == enc.CACHE_FPS:
                features.append(sample)
        profiles[str(fps)] = _video_summary(rows)
    return features, dict(model_id=enc.FACE_ID, backend='CPU XNNPACK', roi=list(enc.ROI), profiles=profiles,
                          unit='ms/turn', decoder='PyAV single source-PTS decode per turn and fps, one thread for decode and RGB conversion',
                          timings_note='roi includes selected-frame RGB conversion and crop; inference excludes init and SRGB wrapping')


def _environment():
    cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
               if line.startswith('model name'))
    return dict(cpu=cpu, cpu_logical=os.cpu_count(), torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(), gpu=torch.cuda.get_device_name(),
                gpu_total_vram_bytes=torch.cuda.get_device_properties(0).total_memory, cuda=torch.version.cuda,
                torch=torch.__version__, transformers=transformers.__version__, mediapipe=mp.__version__,
                pyav=av.__version__, sentence_transformers=sentence_transformers.__version__)


def benchmark(train, turns):
    labels = {role: {e: sum(t.expressions[i] == e for t in turns) for e in CLASSES}
              for i, role in enumerate(ROLES)}
    selection = dict(seed=SEED, turns=len(turns), speakers=len({t.speaker_id for t in turns}),
                     sessions=len({t.session_id for t in turns}), tiers=dict(Counter(t.tier for t in turns)),
                     duration_bands=dict(Counter(_duration_band(t) for t in turns)),
                     dual_high_conf=sum(t.dual_high_conf for t in turns), expressions=labels)
    audio, audio_stats = _audio_benchmark(turns)
    print('WavLM CPU/CUDA benchmark complete', flush=True)
    text, text_stats = _text_benchmark(train, turns)
    print('MiniLM CPU/CUDA benchmark and Train text scan complete', flush=True)
    video, video_stats = _video_benchmark(turns)
    print('Separate 5 fps and 10 fps MediaPipe profiles complete', flush=True)
    return (dict(audio=audio, video=video, text=text),
            dict(environment=_environment(), warmups=WARMUPS, selection=selection,
                 audio=audio_stats, video=video_stats, text=text_stats))


def write_selection(path, turns, video):
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(('turn_id', 'tier', 'speaker_id', 'session_id', 'r01_expr', 'r02_expr',
                         'r01_intensity', 'r02_intensity', 'audio_duration_sec', 'audio_orig_sr',
                         'video_resolution', 'video_orig_fps', 'transcript_char_len'))
        for t, v in zip(turns, video, strict=True):
            writer.writerow((t.turn_id, t.tier, t.speaker_id, t.session_id, *t.expressions, *t.intensities,
                             f'{t.audio_duration:.6f}', t.audio_sample_rate, f"{v['width']}x{v['height']}",
                             f"{v['original_fps']:.6f}", len(t.text)))


def _video_lines(profiles):
    lines = ['## 单档视频对比', '',
             '| fps | 联合有效帧 | 有效率 | 有效帧数值有限 | ROI / 全图回退话轮 | 上脸 Mean Std / Delta | 下脸 Mean Std / Delta |',
             '|---:|---:|---:|---|---:|---:|---:|']
    for fps, p in profiles.items():
        motion = p['dynamics']
        groups = [' / '.join(f'{motion[g][k]["mean"]:.5f}' if motion[g][k]['mean'] is not None else 'N/A'
                             for k in ('mean_std', 'delta')) for g in FACE_GROUPS]
        lines.append(f'| {fps} | {p["valid_frames"]}/{p["frames"]} | {p["valid_fraction"]:.2%} | '
                     f'{"是" if p["valid_values_finite"] else "否"} | {p["crop_modes"].get("roi", 0)} / '
                     f'{p["crop_modes"].get("full_fallback", 0)} | {groups[0]} | {groups[1]} |')
    lines += ['', '按原始 PTS 每轮每档单次解码，新建 VIDEO 跟踪器；ROI 横 20%–80%，纵 0%–45%，'
              '不翻转、不缩放。仅保存 10 fps 主序列。缺失值以 NaN/False 表示，保留原矩阵及合法旋转，Euler `xyz`（度）。',
              '动态统计排除 `_neutral`；对有效帧按系数算标准差或两半均值绝对差，再对系数及可计算话轮等权平均。',
              '', '### 联合有效率低于 85%', '']
    warnings = [(fps, r) for fps, p in profiles.items() for r in p['per_turn'] if r['valid_fraction'] < 0.85]
    if not warnings:
        return [*lines, '无。']
    lines += ['| fps | turn_id | speaker_id | tier | crop_mode | 有效/总帧 | 有效率 |',
              '|---:|---|---|---|---|---:|---:|']
    return lines + [f'| {fps} | {r["turn_id"]} | {r["speaker_id"]} | {r["tier"]} | {r["crop_mode"]} | '
                    f'{r["n_valid"]}/{r["n_frames"]} | {r["valid_fraction"]:.1%} |' for fps, r in warnings]


def _timing_lines(b):
    env, audio, text = b['environment'], b['audio'], b['text']
    stages = [(label, audio['prepare_ms'][key]) for key, label in
              [('read_ms', '音频读取'), ('resample_ms', '音频重采样'), ('read_resample_ms', '读取 + 重采样')]]
    for device in DEVICES:
        for label, result in [('WavLM', audio), ('MiniLM', text)]:
            stages.append((f'{label} {device}', result['profiles'][device]['inference_ms']))
    visual = []
    for fps, profile in b['video']['profiles'].items():
        for key, label in [('decode_ms', '解码'), ('rgb_roi_ms', 'RGB 与 ROI'), ('mediapipe_init_ms', '实例初始化'),
                           ('mediapipe_inference_ms', 'MediaPipe 推理'), ('total_ms', '视觉总耗时')]:
            stage = (f'{fps} fps {label}', profile['timings_ms'][key])
            stages.append(stage)
            if key in ('decode_ms', 'rgb_roi_ms', 'mediapipe_inference_ms'):
                visual.append(stage)
    slowest, timing = max(visual, key=lambda item: item[1]['median'])
    cpu, gpu = (audio['profiles'][d] for d in DEVICES)
    lines = ['## 分环节耗时', '', f'CPU：{env["cpu"]}，逻辑线程 {env["cpu_logical"]}，PyTorch {env["torch_threads"]} 线程；'
             f'GPU：{env["gpu"]}。FP32、CUDA 计时前后同步；MediaPipe CPU XNNPACK。各配置预热 {WARMUPS} 次。', '',
             '| 环节 | 中位数 ms/轮 | P95 ms/轮 |', '|---|---:|---:|']
    lines += [f'| {label} | {v["median"]:.2f} | {v["p95"]:.2f} |' for label, v in stages]
    lines += ['', f'- WavLM RTF 中位数 / P95：CPU {cpu["rtf"]["median"]:.4f} / {cpu["rtf"]["p95"]:.4f}；'
              f'CUDA {gpu["rtf"]["median"]:.4f} / {gpu["rtf"]["p95"]:.4f}。'
              f'CUDA 推理峰值已分配显存 {gpu["peak_allocated_vram_bytes"] / 2**20:.1f} MiB。',
              f'- 最慢视觉子阶段：{slowest}（{timing["median"]:.2f} ms/轮）。两档基准不相加当作单档延迟。']
    return lines


def write_report(path, b):
    s, audio, text = b['selection'], b['audio'], b['text']
    n, scan = s['turns'], text['train_scan']
    lines = ['# Feature Probe-01', '', '## 样本与数值', '',
             f'- 种子 {s["seed"]}。{n} 个 Train 独立话轮，双角色均准入 {s["dual_high_conf"]}/{n}；'
             f'tier：`{s["tiers"]}`；WAV 时长分层：`{s["duration_bands"]}`；{s["speakers"]} 位说话者，{s["sessions"]} 个会话。',
             f'- E00–E07 按角色计数：R01 `{s["expressions"]["R01"]}`；R02 `{s["expressions"]["R02"]}`。',
             f'- 音频：mean/std/tail 均为 `({n}, 13, 768)` float32；有限且有效 {audio["valid_turns"]}/{n}，'
             f'实际收缩端点 {audio["trimmed_turns"]}/{n}，语音短于 1.5 s {audio["short_turns"]}/{n}。'
             '10 ms 能量帧，阈值 max(-60, p95−20) dBFS；仅对超过 0.25 s 的末尾静音按卷积中心裁尾。',
             f'- 文本：`({n}, 384)` float32、L2 归一化；有限 {text["finite_turns"]}/{n}、空文本 {text["sample_empty"]}/{n}、'
             f'截断 {text["sample_truncated"]}/{n}。',
             '- 缓存按 `turn_id` 对齐，角色标签只在 `selection.csv`。仅使用 Train 与输入白名单。', '',
             *_video_lines(b['video']['profiles']), '', '## Train 文本体检', '',
             f'- {scan["turns"]} 个独立话轮：空文本 {scan["empty"]}；非 ASCII {scan["non_ascii_count"]} '
             f'({scan["non_ascii_fraction"]:.2%})；疑似非英文 {scan["suspected_non_english_count"]} '
             f'({scan["suspected_non_english_fraction"]:.2%})。疑似非英文仅以含非拉丁字母为代理。',
             f'- 含特殊词元的 Token 长度均值 / P95 / 最大值：{scan["token_mean"]:.2f} / {scan["token_p95"]:.1f} / '
             f'{scan["token_max"]}；超过 {scan["max_seq_length"]} 上限 {scan["truncated_count"]} 条（{scan["truncated_fraction"]:.2%}）。',
             '', *_timing_lines(b), '', '## 缓存体积', '',
             f'| 缓存 | 实际字节 | 按 1,358/{n} 线性估算字节 |', '|---|---:|---:|']
    lines += [f'| `{key}` | {v["actual"]:,} | {v["projected_1358"]:,} |' for key, v in b['cache_bytes'].items()]
    lines += ['', '线性估算未读取其他划分。ROI 角度未经物理真值校准，RMS 端点未与人工语音边界对照，文本检查不验证领域语义效果。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
