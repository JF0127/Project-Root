import argparse
import json
from pathlib import Path

import torch

from .config import checkpoint_config, load_checkpoint, resolve_device
from .data import ROLES
from .features import EXPRESSION, cached_features, prepare


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('turn_id')
    parser.add_argument('role', choices=ROLES)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    spec, normalization = checkpoint_config(args.checkpoint)
    inputs = prepare(cached_features(args.split, [args.turn_id], spec.modalities), normalization, device)
    inputs['role'] = torch.tensor([ROLES.index(args.role)], device=device)
    out = model(inputs)
    expr, intensity = out['expression_logits'], out['intensity_logits']
    label = int(expr.argmax(dim=-1).item())
    print(json.dumps({'turn_id': args.turn_id, 'role_id': args.role, 'expression': list(EXPRESSION)[label],
                      'intensity': 'medium+' if label and intensity.argmax().item() else 'low',
                      'expression_probabilities': expr.softmax(-1)[0].tolist(),
                      'head_intensity_probabilities': intensity.softmax(-1)[0].tolist()}))


if __name__ == '__main__':
    main()
