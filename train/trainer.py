import csv
import math
import random
from collections import deque
from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch

from .architectures import ModelConfig, build_model
from .config import TrainingConfig, write_json
from .data import model_inputs
from .evaluator import Evaluator, device_batch, loss_curves
from .losses import balance, baseline_loss, loss_terms, mean_losses
from .metrics import save_predictions
from .tracking import digest_config, track_seed


class Trainer:
    def __init__(self, train, val, device, recipe):
        self.train = train
        self.config = TrainingConfig(**recipe['training'])
        self.evaluator = Evaluator(val, device, self.config.batch_turns)
        self.device = device
        self.recipe = recipe
        self.weights = [balance(train.expr[train.expr >= 0], 8)['weights'],
                        balance(train.intensity[train.expr > 0], 2)['weights']]

    def train_epoch(self, model, optimizer, order, weights):
        model.train()
        sums = torch.zeros(4, dtype=torch.float64, device=self.device)
        for batch in self.train.loader(order, self.config.batch_turns):
            batch = device_batch(batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(model_inputs(batch))
            args = (out, batch['expression'], batch['intensity'], weights)
            terms = loss_terms(*args)
            if self.config.selection == 'macro_f1':
                loss = baseline_loss(*args, self.config.intensity_scale)
            else:
                loss = mean_losses(terms, self.config.intensity_scale)[0]
            sums.add_(terms.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            optimizer.step()
        return loss_curves('train', sums, self.config.intensity_scale)

    def fit(self, model, seed, weights, directory, tracker):
        cfg = self.config
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                      betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)
        rng = np.random.default_rng(seed)
        recent = deque(maxlen=cfg.window)
        best, best_epoch, best_raw, min_raw = -math.inf, 0, math.inf, math.inf
        checkpoint = directory / 'best.pt'
        started = perf_counter()
        with (directory / 'history.csv').open('x', newline='', encoding='utf-8') as stream:
            for epoch in range(1, cfg.epochs + 1):
                lr = optimizer.param_groups[0]['lr']
                row = self.train_epoch(model, optimizer, rng.permutation(len(self.train.turn_ids)), weights)
                row.update(self.evaluator.strict_epoch(model, weights, cfg), learning_rate=lr)
                raw = row['val_strict/total_loss']
                recent.append(raw)
                row['val_strict/smoothed_weighted_loss'] = sum(recent) / len(recent)
                score = row['val_strict/expression_macro_f1'] if cfg.selection == 'macro_f1' else -row['val_strict/smoothed_weighted_loss']
                if not math.isfinite(raw) or not math.isfinite(score):
                    raise FloatingPointError('Non-finite validation result')
                min_raw = min(min_raw, raw)
                if score > best:
                    best, best_epoch, best_raw = score, epoch, raw
                    torch.save(model.state_dict(), checkpoint)
                if epoch == 1:
                    writer = csv.DictWriter(stream, fieldnames=['epoch', *row])
                    writer.writeheader()
                writer.writerow({'epoch': epoch, **row})
                stream.flush()
                tracker.epoch(epoch, row)
                scheduler.step()
                if epoch - best_epoch >= cfg.patience:
                    break
        elapsed = perf_counter() - started
        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True), strict=True)
        result = {'best_epoch': best_epoch, 'stopped_epoch': epoch, 'training_seconds': elapsed,
                  'best_raw_val_loss': best_raw, 'min_raw_val_loss': min_raw}
        key = 'best_val_strict_macro_f1' if cfg.selection == 'macro_f1' else 'best_smoothed_val_loss'
        result[key] = best if cfg.selection == 'macro_f1' else -best
        return result

    def run_seed(self, output, study, name, seed, mode, artifact):
        spec = self.recipe['models'][name]
        model_config = ModelConfig(**spec['model'])
        weights = [self.weights[0] if spec['weighted_expression'] else [1.] * 8, self.weights[1]]
        config = {**self.recipe['provenance'], 'model_version': self.recipe['model_version'], 'model_name': name,
                  'model': asdict(model_config), 'model_sha256': digest_config(asdict(model_config)),
                  'input_shapes': {'text': [384], 'audio': [13, 3, 768], 'video': [162]},
                  'dataset': 'Expression v1.1', 'split_scheme': 'scheme1_v1',
                  'train_eligible_records': len(self.train), 'validation_records': len(self.evaluator.val),
                  'training': asdict(self.config), 'optimizer': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8,
                  'scheduler': 'CosineAnnealingLR', 'expression_loss': 'CE / target weight mass',
                  'intensity_loss': 'non-E00 weighted CE sum / non-E00 count',
                  'expression_weights': weights[0], 'intensity_weights': weights[1], 'seed': seed}
        directory = output / name / f'seed_{seed}'
        directory.mkdir(parents=True)
        with track_seed(directory, study, name, seed, config, mode, artifact) as tracker:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = build_model(model_config).to(self.device)
            weights = tuple(torch.tensor(w, device=self.device, dtype=torch.float32) for w in weights)
            result = self.fit(model, seed, weights, directory, tracker)
            probs = self.evaluator.predict(model)
            result.update(self.evaluator.evaluate(probs), experiment=name, seed=seed,
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            if hasattr(model, 'diagnostics'):
                result.update(model.diagnostics())
            write_json(directory / 'metrics.json', result)
            save_predictions(directory / 'val_predictions.parquet', self.evaluator.val, probs)
            tracker.complete(result, directory / 'best.pt', self.recipe['video_zscore'])
        print(f'{name} seed={seed} best_epoch={result["best_epoch"]}', flush=True)
        return result, probs

    def run(self, output, study, mode, artifact=False):
        runs, predictions = {}, {}
        for name in self.recipe['models']:
            runs[name], predictions[name] = [], []
            for seed in self.recipe['seeds']:
                result, probs = self.run_seed(output, study, name, seed, mode, artifact)
                runs[name].append(result)
                predictions[name].append(probs)
        write_json(output / 'summary.json', self.evaluator.summarize(runs, predictions, self.recipe))
