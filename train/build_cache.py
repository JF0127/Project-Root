import argparse
import json
import multiprocessing
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from .data import CACHE, MODELS, ROOT, load_turns
from .encoders import (
    CACHE_FPS,
    AudioEncoder,
    encode_text,
    encode_video,
    face_options,
    load_minilm,
    save_cache,
)

OUTPUT = ROOT / CACHE
CACHE_SPLITS = ('train', 'val')
MODALITY_NAMES = {'audio': '音频', 'video': '视频', 'text': '文本'}
VIDEO_INIT_MESSAGES = (
    b'WARNING: Logging before InitGoogle() is written to STDERR',
    b'INFO: Created TensorFlow Lite XNNPACK delegate for CPU.',
    b'Sets FaceBlendshapesGraph acceleration to xnnpack by default.',
    b'Feedback manager requires a model with a single signature inference. '
    b'Disabling support for feedback tensors.',
)


def resolve_device(request):
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(request)
    if device.type == 'cpu':
        return device, platform.processor() or platform.machine() or 'CPU'
    if device.type != 'cuda':
        raise ValueError('--device must be auto, cpu, cuda or cuda:N')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device, torch.cuda.get_device_name(device)


def _route_worker_stderr(writer):
    os.dup2(writer.fileno(), 2)


def _show_worker_messages(reader):
    with os.fdopen(reader.fileno(), 'rb', closefd=False) as stream:
        for line in stream:
            message = line.rstrip(b'\r\n')
            if not message.endswith(VIDEO_INIT_MESSAGES):
                tqdm.write(message.decode(errors='replace'), file=sys.stderr)


@contextmanager
def video_pool(workers):
    ctx = multiprocessing.get_context('spawn')
    reader, writer = ctx.Pipe(duplex=False)
    with reader:
        relay = Thread(target=_show_worker_messages, args=(reader,))
        relay.start()
        try:
            with writer, ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                             initializer=_route_worker_stderr,
                                             initargs=(writer,)) as pool:
                yield pool
        finally:
            relay.join()


def video_turn(turn):
    return encode_video(turn, face_options(MODELS), fps=CACHE_FPS)


def audio_stats(rows):
    scalars = ('duration_sec', 'speech_end_sec', 'trailing_silence_sec', 'rms_threshold_dbfs')
    return {
        'all_finite': all(r['is_valid'] and np.isfinite([r[k] for k in scalars]).all() for r in rows),
        'trimmed_turns': int(sum(r['speech_end_sec'] < r['duration_sec'] for r in rows)),
        'short_turns': int(sum(r['is_short_turn'] for r in rows)),
    }


def video_stats(rows):
    details = []
    fields = ('frame_blendshapes', 'frame_head_pose', 'frame_matrix_raw', 'frame_rotation')
    for row in rows:
        mask = row['valid_mask']
        frames, valid = len(mask), int(mask.sum())
        details.append(dict(turn_id=row['turn_id'], crop_mode=row['crop_mode'], frames=frames,
                            valid_frames=valid, valid_ratio=valid / frames))
    frames = sum(d['frames'] for d in details)
    valid = sum(d['valid_frames'] for d in details)
    return {
        'fps': CACHE_FPS,
        'valid_frame_values_finite': (
            all(np.isfinite(r[k][r['valid_mask']]).all() for r in rows for k in fields)
            if valid else None
        ),
        'timestamps_finite': all(np.isfinite(r['frame_timestamps']).all() for r in rows),
        'frames': frames,
        'valid_frames': valid,
        'valid_ratio': valid / frames,
        'min_turn_valid_ratio': min(d['valid_ratio'] for d in details),
        'turns_below_0_85': [d for d in details if d['valid_ratio'] < 0.85],
        'full_fallback_turns': [d for d in details if d['crop_mode'] == 'full_fallback'],
    }


def text_stats(rows):
    return {
        'all_finite': all(np.isfinite(r['embedding']).all() for r in rows),
        'empty_turns': int(sum(r['is_empty'] for r in rows)),
        'truncated_turns': int(sum(r['is_truncated'] for r in rows)),
        'max_token_len': int(max(r['token_len'] for r in rows)),
    }


def cache_modality(directory, modality, turns, encode, stats, device, map_turns=map):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = perf_counter()
    rows = list(tqdm(map_turns(encode, turns), total=len(turns), mininterval=0.5,
                     desc=f'{directory.name}/{MODALITY_NAMES[modality]}',
                     bar_format='{desc}：{n_fmt}/{total_fmt} 轮（{percentage:.0f}%），已用 {elapsed}'))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start
    path = save_cache(directory, modality, (t.turn_id for t in turns), rows)
    return dict(device=str(device), cache_size_mb=path.stat().st_size / 1_000_000,
                extraction_seconds=seconds, **stats(rows))


def build(directory, device, device_name, workers):
    turns = load_turns(*CACHE_SPLITS)
    audio = AudioEncoder(MODELS)
    audio.model.to(device)
    text = partial(encode_text, model=load_minilm(MODELS).to(device))
    summary = {}
    with video_pool(workers) as pool:
        for split in CACHE_SPLITS:
            subset = [t for t in turns if t.split == split]
            out = directory / split
            out.mkdir()
            summary[split] = {
                'turns': len(subset), 'audio_text_device_name': device_name, 'video_workers': workers,
                'audio': cache_modality(out, 'audio', subset, audio.encode, audio_stats, device),
                'video': cache_modality(out, 'video', subset, video_turn, video_stats,
                                       torch.device('cpu'), map_turns=pool.map),
                'text': cache_modality(out, 'text', subset, text, text_stats, device),
            }
    (directory / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto', metavar='auto|cpu|cuda[:N]')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()
    if os.path.lexists(OUTPUT):
        raise FileExistsError(f'Refusing to overwrite {OUTPUT}')
    if args.workers < 1:
        parser.error('--workers must be positive')
    device, name = resolve_device(args.device)
    print(f'音频/文本：{device}（{name}）；视频：CPU，{args.workers} 个进程', flush=True)
    with TemporaryDirectory(dir=ROOT, prefix=f'.{OUTPUT.name}-') as temp:
        stage = Path(temp)
        build(stage, device, name, args.workers)
        OUTPUT.mkdir()
        for entry in (*CACHE_SPLITS, 'summary.json'):
            (stage / entry).rename(OUTPUT / entry)
    print(f'缓存完成：{OUTPUT}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n缓存构建已中断。', file=sys.stderr)
        raise SystemExit(130) from None
