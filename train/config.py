import json
from dataclasses import dataclass

import torch

from .architectures import ModelConfig, build_model


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    patience: int
    selection: str
    window: int = 1
    batch_turns: int = 32
    lr: float = 0.001
    weight_decay: float = 0.01
    eta_min: float = 1e-5
    max_grad_norm: float = 1.0
    intensity_scale: float = 0.5

    def __post_init__(self):
        if min(self.epochs, self.patience, self.window, self.batch_turns) < 1:
            raise ValueError('Epoch, patience, window and batch sizes must be positive')
        if self.selection not in ('macro_f1', 'smoothed_weighted_loss'):
            raise ValueError('Unknown checkpoint selection criterion')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def checkpoint_config(path):
    portable = path.with_name('inference_config.json')
    if portable.is_file():
        config = json.loads(portable.read_text(encoding='utf-8'))
        return ModelConfig(**config['model']), config['video_zscore']
    run = json.loads((path.parents[2] / 'config.json').read_text(encoding='utf-8'))
    name = path.parents[1].name
    if 'models' in run:
        return ModelConfig(**run['models'][name]['model']), run['video_zscore']
    spec = run['experiments'][name]
    arch = run['architecture']
    projection = {None: 'original', 'original_2304_to_128': 'original',
                  'shared_768_to_64_then_192_to_128': 'shared'}[spec.get('audio_projection')]
    model = ModelConfig(tuple(spec['modalities']), spec['conditioning'], projection, arch['dropout'],
                        arch['hidden_dim'], arch['role_dim'])
    return model, run['video'] if 'video' in run else run['data']['video_zscore']


def load_checkpoint(path, device):
    config, _ = checkpoint_config(path)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.eval()


def resolve_device(request):
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(request)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('Expected cpu, cuda or cuda:N')
    return device
