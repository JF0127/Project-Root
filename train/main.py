import argparse
import json
import os
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from .architectures import ModelConfig
from .config import TrainingConfig, resolve_device, write_json
from .data import ROOT
from .features import load_data
from .tracking import provenance
from .trainer import Trainer


def recipe_for(path, names, seeds):
    recipe = json.loads(path.read_text(encoding='utf-8'))
    names = names or list(recipe['models'])
    if not names or len(set(names)) != len(names) or not set(names) <= recipe['models'].keys():
        raise ValueError('Expected distinct model names from the recipe')
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in names):
        raise ValueError('Invalid model name')
    recipe['models'] = {name: recipe['models'][name] for name in names}
    for spec in recipe['models'].values():
        spec['model'] = asdict(ModelConfig(**spec['model']))
        if not isinstance(spec['weighted_expression'], bool):
            raise ValueError('weighted_expression must be boolean')
    recipe['training'] = asdict(TrainingConfig(**recipe['training']))
    recipe['seeds'] = seeds or recipe['seeds']
    if not recipe['seeds'] or len(set(recipe['seeds'])) != len(recipe['seeds']):
        raise ValueError('Expected distinct seeds')
    return recipe


def launch_config():
    studies = ('baseline', 'audio_compression', 'regularization', 'fusion')
    modes = ('online', 'offline', 'disabled')
    parser = argparse.ArgumentParser()
    parser.add_argument('study', nargs='?', choices=studies)
    parser.add_argument('--run')
    parser.add_argument('--config', type=Path, help='defaults to train/config.json, or the named study recipe')
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--device')
    parser.add_argument('--swanlab-mode', choices=modes)
    parser.add_argument('--model-artifact', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    default = ROOT / 'experiments' / args.study / 'recipe.json' if args.study else Path(__file__).with_name('config.json')
    recipe = recipe_for(args.config or default, args.experiments, args.seeds)
    args.study = args.study or recipe.get('study')
    args.device = args.device or recipe.get('device', 'auto')
    args.swanlab_mode = args.swanlab_mode or recipe.get('swanlab_mode', 'offline')
    if args.model_artifact is None:
        args.model_artifact = recipe.get('model_artifact', False)
    if args.study not in studies or args.swanlab_mode not in modes or not isinstance(args.model_artifact, bool):
        parser.error('Invalid study, swanlab_mode or model_artifact in config')
    if args.model_artifact and args.swanlab_mode == 'disabled':
        parser.error('Artifacts require offline or online tracking')
    args.run = args.run or recipe.get('run') or datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    if not re.fullmatch(r'[\w-]+', args.run):
        parser.error('--run must be a new directory name')
    directory = ROOT / 'experiments' / args.study
    output = directory / args.run
    if os.path.lexists(output):
        raise FileExistsError(f'Refusing to overwrite {output}')
    recipe['run'] = args.run
    return args, recipe, output


def main():
    args, recipe, output = launch_config()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    train, val, normalization = load_data('cpu')
    device = resolve_device(args.device)
    recipe.update(video_zscore=normalization, provenance=provenance(), study=args.study,
                  device=str(device), torch_version=str(torch.__version__), swanlab_mode=args.swanlab_mode,
                  model_artifact=args.model_artifact)
    trainer = Trainer(train, val, device, recipe)
    recipe.update(expression_weights=trainer.weights[0], non_neutral_intensity_weights=trainer.weights[1])
    output.mkdir(parents=True)
    write_json(output / 'config.json', recipe)
    trainer.run(output, args.study, args.swanlab_mode, args.model_artifact)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
