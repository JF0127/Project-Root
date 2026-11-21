import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import metrics
from .config import checkpoint_config, load_checkpoint, resolve_device
from .data import model_inputs
from .features import load_validation, val_subsets
from .losses import loss_terms, mean_losses


def device_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def loss_curves(prefix, terms, scale):
    names = ('total_loss', 'expression_loss', 'intensity_loss')
    return {f'{prefix}/{key}': value.item() for key, value in zip(names, mean_losses(terms, scale), strict=True)}


class Evaluator:
    def __init__(self, val, device, batch_turns=32):
        self.val = val
        self.device = device
        self.batch_turns = batch_turns
        self.subsets = val_subsets(val)

    @torch.inference_mode()
    def predict(self, model):
        model.eval()
        expr, intensity = [], []
        for batch in self.val.loader(self.subsets['val_all'], self.batch_turns):
            out = model(model_inputs(device_batch(batch, self.device)))
            expr.append(out['expression_logits'].softmax(-1).cpu().numpy().reshape(-1, 2, 8))
            intensity.append(out['intensity_logits'].softmax(-1).cpu().numpy().reshape(-1, 2, 2))
        return metrics.Probabilities(np.concatenate(expr), np.concatenate(intensity))

    @torch.inference_mode()
    def strict_epoch(self, model, weights, config):
        model.eval()
        indices = self.subsets['val_strict']
        size = self.batch_turns if config.selection == 'macro_f1' else len(indices)
        weighted, unweighted, pred, raw_int = [], [], [], []
        for batch in self.val.loader(indices, size):
            batch = device_batch(batch, self.device)
            out = model(model_inputs(batch))
            args = (out, batch['expression'], batch['intensity'])
            weighted.append(loss_terms(*args, weights))
            unweighted.append(loss_terms(*args, (None, None)))
            expr = out['expression_logits']
            if config.selection == 'macro_f1':
                expr = expr.softmax(-1)
            pred.append(expr.argmax(-1).cpu().numpy())
            raw_int.append(out['intensity_logits'].argmax(-1).cpu().numpy())
        truth = self.val.expr[indices].ravel()
        active = truth != 0
        expression = metrics.expression_metrics(truth, np.concatenate(pred))
        cm = metrics.confusion(self.val.intensity[indices].ravel()[active], np.concatenate(raw_int)[active], 2)
        return {**loss_curves('val_strict', torch.stack(weighted).sum(0), config.intensity_scale),
                **loss_curves('val_strict_unweighted', torch.stack(unweighted).sum(0), config.intensity_scale),
                'val_strict/expression_macro_f1': expression['macro_f1'],
                'val_strict/intensity_macro_f1': float(metrics.class_f1(cm).mean()),
                **{f'val_strict/{label}/{stat}': scores[stat] for label, scores in expression['per_class'].items()
                   for stat in ('recall', 'support')}}

    def evaluate(self, probabilities):
        result = {}
        for key, ix in self.subsets.items():
            expr = self.val.expr[ix]
            probs = probabilities.subset(ix)
            scores = metrics.evaluate(expr, self.val.intensity[ix], probs)
            expression = scores['expression']
            expression['macro_f1_excl_E05'] = float(metrics.macro_f1_excl_e05(metrics.class_f1(np.asarray(expression['confusion_matrix']))))
            pred = probs.expr.argmax(axis=-1)
            cm = metrics.confusion(expr[:, 0] != expr[:, 1], pred[:, 0] != pred[:, 1], 2)
            scores['paired'].update(tp=int(cm[1, 1]), fn=int(cm[1, 0]), fp=int(cm[0, 1]))
            result[key] = scores
        return result

    def summarize(self, runs, predictions, recipe):
        results = {}
        for name, members in predictions.items():
            ensemble = self.evaluate(metrics.probability_mean(members))
            single = {key: metrics.summarize_seeds(runs[name], key) for key in self.subsets}
            for key in self.subsets:
                matrices = np.array([r[key]['expression']['confusion_matrix'] for r in runs[name]])
                single[key]['expression.confusion_matrix'] = {'mean': matrices.mean(0).tolist(),
                    'std': matrices.std(0, ddof=1).tolist() if len(matrices) > 1 else None}
            results[name] = {'ensemble_probability': ensemble, 'single_model_mean_std': single}
            if 'wavlm_layer_weights' in runs[name][0]:
                layers = {str(r['seed']): r['wavlm_layer_weights'] for r in runs[name]}
                results[name]['wavlm_layer_weights'] = {'by_seed': layers, 'mean': np.mean(list(layers.values()), axis=0).tolist()}
        bootstrap = metrics.paired_bootstrap(self.val, self.subsets, predictions, comparisons=recipe['comparisons'],
                                             metrics=recipe['bootstrap_metrics'], seeds=recipe['seeds'])
        return {'experiments': results, 'seeds': recipe['seeds'], 'std_ddof': 1,
                'paired_bootstrap': {'B': metrics.BOOTSTRAP_B, 'seed': metrics.BOOTSTRAP_SEED, **bootstrap}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = resolve_device(args.device)
    _, normalization = checkpoint_config(args.checkpoint)
    evaluator = Evaluator(load_validation(normalization, 'cpu'), device)
    print(json.dumps(evaluator.evaluate(evaluator.predict(load_checkpoint(args.checkpoint, device))), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
