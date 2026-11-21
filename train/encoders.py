import warnings
from contextlib import nullcontext

import av
import mediapipe as mp
import numpy as np
import soundfile as sf
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation
from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

WAVLM_ID = 'wavlm-base-plus'
MINILM_ID = 'all-MiniLM-L6-v2'
FACE_ID = 'mediapipe/face_landmarker.task'
RATE = 16000
CACHE_FPS = 10
ROI = (0.20, 0.0, 0.80, 0.45)
EULER_ORDER = 'xyz'
BLENDSHAPES = (
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
    'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
    'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
)


def _use_ieee_fp32():
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'


def _endpoint(wave, duration):
    starts = np.arange(0, len(wave), RATE // 100)
    ends = np.minimum(starts + RATE // 100, len(wave))
    energy = np.add.reduceat(np.square(wave, dtype=np.float64), starts) / (ends - starts)
    levels = 10 * np.log10(np.maximum(energy, 1e-24))
    threshold = max(-60.0, float(np.percentile(levels, 95)) - 20.0)
    active = np.flatnonzero(levels >= threshold)
    last = min(duration, ends[active[-1]] / RATE) if active.size else 0.0
    silence = duration - last
    return (last if silence > 0.25 else duration), silence, threshold


def _pool_audio(states, wave, turn, hop, receptive):
    steps = states[0].shape[1]
    if len(states) != 13 or any(s.shape != (1, steps, 768) for s in states):
        raise ValueError(f'Unexpected WavLM output: {[tuple(s.shape) for s in states]}')
    if not steps or steps != (len(wave) - receptive) // hop + 1:
        raise ValueError(f'Unexpected hidden time axis: {turn.turn_id}')
    centers = (np.arange(steps) * hop + (receptive - 1) / 2) / RATE
    end, trailing, threshold = _endpoint(wave, turn.audio_duration)
    stop = max(1, int(np.searchsorted(centers, end, side='right')))
    tail = min(stop - 1, int(np.searchsorted(centers, max(0, end - 1.5))))
    with torch.inference_mode():
        hidden = torch.stack([s[0] for s in states])
        mean = hidden[:, :stop].mean(dim=1)
        std = hidden[:, :stop].std(dim=1, correction=0)
        tail_mean = hidden[:, tail:stop].mean(dim=1)
        mean, std, tail_mean = (x.cpu().numpy() for x in (mean, std, tail_mean))
    return dict(turn_id=turn.turn_id, hidden_mean=mean, hidden_std=std, hidden_tail=tail_mean,
                orig_sample_rate=turn.audio_sample_rate, duration_sec=turn.audio_duration,
                speech_end_sec=end, trailing_silence_sec=trailing, is_short_turn=end < 1.5,
                rms_threshold_dbfs=threshold, hidden_steps=steps, speech_hidden_stop=stop,
                tail_hidden_start=tail, is_valid=all(np.isfinite(x).all() for x in (mean, std, tail_mean)))


class AudioEncoder:
    def __init__(self, models):
        _use_ieee_fp32()
        path = models / WAVLM_ID
        self.model = WavLMModel.from_pretrained(path, local_files_only=True).float().eval()
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
        cfg, ext = self.model.config, self.extractor
        if (cfg.hidden_size, cfg.num_hidden_layers, ext.sampling_rate, ext.feature_size,
                ext.do_normalize, cfg.add_adapter) != (768, 12, RATE, 1, False, False):
            raise ValueError('Unexpected WavLM model or feature extractor configuration')
        hop, receptive = 1, 1
        for layer, kernel, stride in zip(self.model.feature_extractor.conv_layers,
                                         cfg.conv_kernel, cfg.conv_stride, strict=True):
            conv = layer.conv
            if (conv.kernel_size, conv.stride, conv.padding, conv.dilation) != (
                    (kernel,), (stride,), (0,), (1,)):
                raise ValueError('WavLM convolution geometry differs from config')
            receptive += (kernel - 1) * hop
            hop *= stride
        self.config = dict(hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
                           feature_extractor=ext.to_dict(), conv_kernel=cfg.conv_kernel,
                           conv_stride=cfg.conv_stride, hop=hop, receptive=receptive)

    def prepare(self, turn, stage=nullcontext):
        with stage('read'):
            pcm, rate = sf.read(turn.audio, dtype='float32', always_2d=True)
        with stage('resample'):
            wave = pcm.mean(axis=1, dtype=np.float32)
            if rate != RATE:
                wave = resample_poly(wave, RATE, rate).astype(np.float32, copy=False)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid waveform: {turn.turn_id}')
        batch = self.extractor(wave, sampling_rate=RATE, padding=False, return_tensors='pt')
        return wave, batch

    def encode(self, turn, *, prepared=None, stage=nullcontext):
        wave, batch = self.prepare(turn, stage) if prepared is None else prepared
        inputs = {key: tensor.to(self.model.device) for key, tensor in batch.items()}
        with stage('inference'), torch.inference_mode():
            states = self.model(**inputs, output_hidden_states=True).hidden_states
        return _pool_audio(states, wave, turn, self.config['hop'], self.config['receptive'])


def load_minilm(models):
    _use_ieee_fp32()
    model = SentenceTransformer(str(models / MINILM_ID), device='cpu', local_files_only=True)
    if model.max_seq_length != 256 or model.get_embedding_dimension() != 384:
        raise ValueError('Unexpected MiniLM limits')
    return model.float().eval()


def token_lengths(model, texts):
    return model.tokenizer(texts, truncation=False, add_special_tokens=True,
                           return_length=True, verbose=False)['length']


def encode_text(turn, model, *, token_len=None, stage=nullcontext):
    with stage('inference'):
        vector = model.encode([turn.text], normalize_embeddings=True, precision='float32',
                              show_progress_bar=False)[0]
    if vector.shape != (384,) or vector.dtype != np.float32:
        raise ValueError('Unexpected MiniLM output')
    if token_len is None:
        token_len, = token_lengths(model, [turn.text])
    return dict(turn_id=turn.turn_id, embedding=vector, char_len=len(turn.text), token_len=token_len,
                is_truncated=token_len > model.max_seq_length, is_empty=not turn.text.strip())


def face_options(models):
    return vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(models / FACE_ID),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)


def _decode(path, fps):
    frames, pts = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        base = stream.time_base
        width, height, original_fps = stream.width, stream.height, float(stream.average_rate)
        origin, previous, target = None, None, 0
        for frame in container.decode(stream):
            p = frame.pts
            if p is None or frame.time_base != base or (previous is not None and p < previous):
                raise ValueError(f'Invalid source PTS: {path}')
            if origin is None:
                origin = p
            relative = (p - origin) * base.numerator * fps
            if relative >= target * base.denominator:
                frames.append(frame)
                pts.append(p)
                target = relative // base.denominator + 1
            previous = p
    if not pts:
        raise ValueError(f'Empty video: {path}')
    return frames, np.array(pts, np.int64), base, width, height, original_fps


def _timestamps_ms(pts, base):
    timestamps = []
    for p in pts:
        scaled = (int(p) - int(pts[0])) * base.numerator * 1000
        rounded = (2 * scaled + base.denominator) // (2 * base.denominator)
        timestamps.append(max(rounded, timestamps[-1] + 1 if timestamps else 0))
    return np.array(timestamps, np.int64)


def _head_pose(matrix):
    a = matrix[:3, :3].astype(np.float64)
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4, rtol=0):
        return None, None
    scale = np.linalg.norm(a) / np.sqrt(3)
    if not 1e-6 <= scale <= 1e6:
        return None, None
    q = a / scale
    if not np.allclose(q.T @ q, np.eye(3), atol=2e-3, rtol=0) or abs(np.linalg.det(q) - 1) > 2e-3:
        return None, None
    u, _, vt = np.linalg.svd(q)
    rotation = u @ vt
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        euler = Rotation.from_matrix(rotation).as_euler(EULER_ORDER, degrees=True)
    for warning in caught:
        if 'Gimbal lock' in str(warning.message):
            return rotation, None
        warnings.warn(warning.message, warning.category, stacklevel=2)
    return rotation, euler


def _face_arrays(results):
    n = len(results)
    blend = np.full((n, len(BLENDSHAPES)), np.nan, np.float32)
    pose = np.full((n, 3), np.nan, np.float32)
    raw = np.full((n, 4, 4), np.nan, np.float32)
    rotation = np.full((n, 3, 3), np.nan, np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, result in enumerate(results):
        if not result.facial_transformation_matrixes:
            continue
        matrix, = result.facial_transformation_matrixes
        matrix = np.asarray(matrix, np.float32)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            continue
        raw[i] = matrix
        proper, euler = _head_pose(matrix)
        if proper is not None:
            rotation[i] = proper
        if euler is None or not result.face_blendshapes:
            continue
        categories, = result.face_blendshapes
        values = {c.category_name: c.score for c in categories}
        if len(categories) != 52 or values.keys() != set(BLENDSHAPES):
            raise ValueError('Unexpected MediaPipe blendshape schema')
        scores = np.array([values[key] for key in BLENDSHAPES], np.float32)
        if np.isfinite(scores).all() and np.isfinite(euler).all():
            blend[i], pose[i], valid[i] = scores, euler, True
    return dict(frame_blendshapes=blend, frame_head_pose=pose, frame_matrix_raw=raw,
                frame_rotation=rotation, valid_mask=valid)


def _track(frames, reformatter, bounds, timestamps, options, stage):
    with stage('fallback_rgb' if bounds is None else 'rgb_roi'):
        images = []
        for frame in frames:
            rgb = reformatter.reformat(frame, format='rgb24', threads=1).to_ndarray()
            if bounds is not None:
                left, top, right, bottom = bounds
                rgb = rgb[top:bottom, left:right]
            images.append(np.ascontiguousarray(rgb))
    with stage('mediapipe_init'):
        tracker = vision.FaceLandmarker.create_from_options(options)
    results = []
    with tracker:
        for rgb, timestamp in zip(images, timestamps, strict=True):
            with stage('srgb_wrap'):
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with stage('mediapipe_inference'):
                results.append(tracker.detect_for_video(image, int(timestamp)))
    return _face_arrays(results)


def encode_video(turn, options, fps=CACHE_FPS, *, stage=nullcontext):
    if not isinstance(fps, (int, np.integer)) or fps <= 0:
        raise ValueError('Video fps must be a positive integer')
    with stage('decode'):
        frames, pts, base, width, height, original_fps = _decode(turn.video, fps)
    timestamps = _timestamps_ms(pts, base)
    reformatter = av.video.reformatter.VideoReformatter()
    bounds = tuple(round(v * d) for v, d in zip(ROI, (width, height, width, height), strict=True))
    arrays = _track(frames, reformatter, bounds, timestamps, options, stage)
    crop_mode = 'roi'
    if not arrays['valid_mask'].any():
        arrays = _track(frames, reformatter, None, timestamps, options, stage)
        crop_mode = 'full_fallback'
    return dict(**arrays, turn_id=turn.turn_id, fps=fps, frame_pts=pts,
                frame_timestamps=(pts - pts[0]) * float(base), frame_mediapipe_ms=timestamps,
                time_base_num=base.numerator, time_base_den=base.denominator, crop_mode=crop_mode,
                width=width, height=height, original_fps=original_fps)


AUDIO_FIELDS = ('hidden_mean', 'hidden_std', 'hidden_tail', 'orig_sample_rate', 'duration_sec',
                'speech_end_sec', 'trailing_silence_sec', 'is_short_turn', 'rms_threshold_dbfs',
                'hidden_steps', 'speech_hidden_stop', 'tail_hidden_start', 'is_valid')
TEXT_FIELDS = ('embedding', 'char_len', 'token_len', 'is_truncated', 'is_empty')
VIDEO_FRAME_FIELDS = ('frame_blendshapes', 'frame_head_pose', 'frame_matrix_raw', 'frame_rotation',
                      'valid_mask', 'frame_timestamps', 'frame_pts', 'frame_mediapipe_ms')
VIDEO_TURN_FIELDS = ('time_base_num', 'time_base_den', 'crop_mode')


def save_cache(directory, modality, turn_ids, rows):
    ids, rows = list(turn_ids), list(rows)
    if not ids or len(set(ids)) != len(ids) or ids != [r['turn_id'] for r in rows]:
        raise ValueError(f'{modality} rows must match the ordered unique turn IDs')
    if modality == 'video':
        fps, = {r['fps'] for r in rows}
        ends = np.cumsum([len(r['valid_mask']) for r in rows], dtype=np.int64)
        arrays = {k: np.concatenate([r[k] for r in rows]) for k in VIDEO_FRAME_FIELDS}
        arrays.update(model_id=FACE_ID, fps=np.int64(fps), roi=np.array(ROI), euler_order=EULER_ORDER,
                      turn_offsets=np.column_stack((np.r_[0, ends[:-1]], ends)),
                      blendshape_names=np.array(BLENDSHAPES), neutral_index=BLENDSHAPES.index('_neutral'),
                      coord_system='mediapipe_camera_right_handed', horizontal_flip=False)
        fields = VIDEO_TURN_FIELDS
    elif modality == 'audio':
        arrays = dict(model_id=WAVLM_ID, target_sample_rate=RATE)
        fields = AUDIO_FIELDS
    elif modality == 'text':
        arrays = dict(model_id=MINILM_ID)
        fields = TEXT_FIELDS
    else:
        raise ValueError(f'Unknown cache modality: {modality}')
    arrays.update({k: np.stack([r[k] for r in rows]) for k in fields})
    path = directory / f'{modality}_features.npz'
    np.savez_compressed(path, allow_pickle=False, turn_id=np.array(ids), **arrays)
    return path
