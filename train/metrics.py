from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

from .data import ROLES
from .features import EXPRESSION

SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
COMPARISONS = (('M0', 'M0_CE'), ('M1', 'M0'), ('M2', 'M0'), ('M3', 'M0'),
               ('M3', 'M3_no_role'), ('M3', 'M3_concat'))
BOOTSTRAP_METRICS = ('expression.macro_f1', 'paired.disagreement.accuracy',
                     'non_neutral_intensity.macro_f1')


@dataclass
class Probabilities:
    expr: np.ndarray
    intensity: np.ndarray

    def subset(self, indices):
        return Probabilities(self.expr[indices], self.intensity[indices])

    def decisions(self):
        expr, raw = self.expr.argmax(axis=-1), self.intensity.argmax(axis=-1)
        return expr, raw, np.where(expr == 0, 0, raw)


def confusion(truth, pred, classes):
    return np.bincount((classes * truth + pred).ravel(), minlength=classes**2).reshape(classes, classes)


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=denominator != 0)


def class_f1(cm):
    return ratio(2 * cm.diagonal(), cm.sum(axis=0) + cm.sum(axis=1))


def positive_scores(cm):
    return {'precision': float(ratio(cm[1, 1], cm[:, 1].sum())),
            'recall': float(ratio(cm[1, 1], cm[1].sum())), 'f1': float(class_f1(cm)[1])}


def expression_metrics(truth, pred):
    cm = confusion(truth, pred, 8)
    support = cm.sum(axis=1)
    precision, recall = ratio(cm.diagonal(), cm.sum(axis=0)), ratio(cm.diagonal(), support)
    f1 = class_f1(cm)
    return {
        'macro_f1': float(f1.mean()), 'active_macro_f1': float(f1[support > 0].mean()),
        'weighted_f1': float(np.average(f1, weights=support)), 'accuracy': float(cm.trace() / cm.sum()),
        'per_class': {label: {'precision': float(precision[i]), 'recall': float(recall[i]),
                              'f1': float(f1[i]), 'support': int(support[i])}
                      for label, i in EXPRESSION.items()},
        'missing_classes': [label for label, i in EXPRESSION.items() if support[i] == 0],
        'confusion_matrix': cm.tolist(),
    }


def auroc(truth, score):
    positive = truth == 1
    n_pos = int(positive.sum())
    n_neg = truth.size - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = rankdata(score, method='average')[positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def non_neutral_metrics(truth, raw, probability):
    cm = confusion(truth, raw, 2)
    return {'records': int(truth.size), 'macro_f1': float(class_f1(cm).mean()),
            **{f'medium_plus_{key}': value for key, value in positive_scores(cm).items()},
            'auroc': auroc(truth, probability)}


def stratum_metrics(truth, pred):
    cm = confusion(truth, pred, 8)
    return {'turns': len(truth), 'accuracy': float(cm.trace() / cm.sum()) if cm.sum() else None,
            'macro_f1': float(class_f1(cm).mean())}


def intensity_order(truth, pred):
    delta = pred[:, 1] - pred[:, 0]
    cm = confusion(truth[:, 1] > truth[:, 0], delta > 0, 2)
    result = {'turns': len(truth), 'r02_gt_r01_f1': float(class_f1(cm)[1])}
    for key, mask in (('r02_gt_r01', delta > 0), ('r02_eq_r01', delta == 0), ('r02_lt_r01', delta < 0)):
        result[key] = {'count': int(mask.sum()), 'rate': float(mask.mean()) if mask.size else None}
    return result


def paired_metrics(expr, intensity, pred_expr, final_intensity):
    different, pred_diff = expr[:, 0] != expr[:, 1], pred_expr[:, 0] != pred_expr[:, 1]
    same_non_neutral = ~different & (expr[:, 0] != 0)
    diff = positive_scores(confusion(different, pred_diff, 2))
    return {
        'paired_expr_exact_match': float((expr == pred_expr).all(axis=1).mean()),
        'disagreement': stratum_metrics(expr[different], pred_expr[different]),
        'agreement': stratum_metrics(expr[~different], pred_expr[~different]),
        **{f'diff_{k}': v for k, v in diff.items()},
        'predicted_disagreement_rate': float(pred_diff.mean()),
        'same_non_neutral_intensity_order': intensity_order(intensity[same_non_neutral],
                                                            final_intensity[same_non_neutral]),
    }


def evaluate(expr, intensity, probabilities):
    pred, raw, final = probabilities.decisions()
    active = expr != 0
    return {
        'turns': len(expr), 'records': int(expr.size), 'expression': expression_metrics(expr, pred),
        'non_neutral_intensity': non_neutral_metrics(intensity[active], raw[active],
                                                     probabilities.intensity[..., 1][active]),
        'end_to_end': {'joint_exact_acc': float(((expr == pred) & (intensity == final)).mean()),
                       'overall_intensity_acc': float((intensity == final).mean()),
                       'overall_intensity_macro_f1': float(class_f1(confusion(intensity, final, 2)).mean())},
        'paired': paired_metrics(expr, intensity, pred, final),
    }


def save_predictions(path, val, probabilities):
    pred, raw, final = probabilities.decisions()
    expr_names, int_names = np.array(list(EXPRESSION)), np.array(['low', 'medium+'])
    columns = {
        'turn_id': np.repeat(val.turn_ids, 2), 'role_id': np.tile(ROLES, len(val.turn_ids)),
        'tier': np.repeat(val.tiers, 2), 'true_expression': expr_names[val.expr.ravel()],
        'pred_expression': expr_names[pred.ravel()], 'true_intensity': int_names[val.intensity.ravel()],
        'head_pred_intensity': int_names[raw.ravel()], 'pred_intensity': int_names[final.ravel()],
        'expr_probabilities': pa.FixedSizeListArray.from_arrays(pa.array(probabilities.expr.ravel()), 8),
        'intensity_probabilities': pa.FixedSizeListArray.from_arrays(pa.array(probabilities.intensity.ravel()), 2),
    }
    pq.write_table(pa.table(columns), path)


def turn_resamples(n):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(n, size=(BOOTSTRAP_B, n))
    counts = np.zeros((BOOTSTRAP_B + 1, n), dtype=np.float64)
    counts[0] = 1
    np.add.at(counts, (np.arange(1, BOOTSTRAP_B + 1)[:, None], draws), 1)
    return counts


def resampled_class_f1(counts, truth, pred, classes, keep=None):
    labels = np.arange(classes)
    actual, predicted = truth[..., None] == labels, pred[..., None] == labels
    if keep is not None:
        actual &= keep[..., None]
        predicted &= keep[..., None]
    twice_hits = 2 * (actual & predicted).sum(axis=1)
    totals = actual.sum(axis=1) + predicted.sum(axis=1)
    return ratio(counts @ twice_hits, counts @ totals)


def resampled_macro(counts, truth, pred, classes, keep=None):
    return resampled_class_f1(counts, truth, pred, classes, keep).mean(axis=1)


def macro_f1_excl_e05(per_class_f1):
    return np.delete(per_class_f1, EXPRESSION['E05'], axis=-1).mean(axis=-1)


def resampled_scores(counts, expr, intensity, probabilities, metrics=BOOTSTRAP_METRICS):
    pred, raw, final = probabilities.decisions()
    disagree = expr[:, 0] != expr[:, 1]
    size = counts @ (2 * disagree)
    hits = counts @ (disagree * (expr == pred).sum(axis=1))
    acc = np.divide(hits, size, out=np.full_like(size, np.nan), where=size != 0)
    expr_f1 = resampled_class_f1(counts, expr, pred, 8)
    scores = {'expression.macro_f1': expr_f1.mean(axis=1), 'paired.disagreement.accuracy': acc,
              'non_neutral_intensity.macro_f1': resampled_macro(counts, intensity, raw, 2, expr != 0)}
    if 'expression.macro_f1_excl_E05' in metrics:
        scores['expression.macro_f1_excl_E05'] = macro_f1_excl_e05(expr_f1)
    if 'end_to_end.joint_exact_acc' in metrics:
        joint_hits = ((expr == pred) & (intensity == final)).sum(axis=1)
        scores['end_to_end.joint_exact_acc'] = (counts @ joint_hits) / (2 * counts.sum(axis=1))
    return np.column_stack([scores[key] for key in metrics])


def probability_mean(members):
    return Probabilities(np.mean([p.expr for p in members], axis=0, dtype=np.float64),
                         np.mean([p.intensity for p in members], axis=0, dtype=np.float64))


def interval(delta):
    draws = delta[1:]
    valid = draws[np.isfinite(draws)]
    return {'delta': float(delta[0]) if np.isfinite(delta[0]) else None,
            'ci95': np.percentile(valid, [2.5, 97.5]).tolist() if valid.size else [None, None],
            'valid_resamples': int(valid.size)}


def contrast(delta, metrics=BOOTSTRAP_METRICS, seeds=SEEDS):
    ensemble, paired_seeds = delta[0], delta[1:]
    seed_mean = paired_seeds.mean(axis=0)
    return {metric: {'ensemble_probability_delta': interval(ensemble[:, i]),
                     'mean_seed_delta': interval(seed_mean[:, i]),
                     'per_seed_delta': {str(seed): float(d) if np.isfinite(d) else None
                                        for seed, d in zip(seeds, paired_seeds[:, 0, i], strict=True)}}
            for i, metric in enumerate(metrics)}


def paired_bootstrap(val, subsets, predictions, *, comparisons=COMPARISONS, metrics=BOOTSTRAP_METRICS, seeds=SEEDS):
    pairs = [(a, b) for a, b in comparisons if a in predictions and b in predictions]
    needed = {name for pair in pairs for name in pair}
    result = {}
    for subset, ix in subsets.items():
        counts = turn_resamples(len(ix))
        scores = {}
        for name in needed:
            members = [p.subset(ix) for p in predictions[name]]
            scores[name] = np.stack([resampled_scores(counts, val.expr[ix], val.intensity[ix], p, metrics)
                                     for p in (probability_mean(members), *members)])
        result[subset] = {f'{a} - {b}': contrast(scores[a] - scores[b], metrics, seeds) for a, b in pairs}
    return result


def scalar_fields(tree, prefix=''):
    result = {}
    for key, value in tree.items():
        path = prefix + key
        if isinstance(value, dict):
            result.update(scalar_fields(value, path + '.'))
        elif isinstance(value, int | float) or value is None:
            result[path] = value
    return result


def mean_std(values):
    valid = [v for v in values if v is not None]
    return {'mean': float(np.mean(valid)) if valid else None,
            'std': float(np.std(valid, ddof=1)) if len(valid) > 1 else None}


def summarize_seeds(metrics, subset):
    rows = [scalar_fields(m[subset]) for m in metrics]
    return {key: mean_std([row[key] for row in rows]) for key in rows[0]}
