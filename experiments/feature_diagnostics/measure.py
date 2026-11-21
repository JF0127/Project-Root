import json
import os
import tempfile
from pathlib import Path

from train.data import ROOT, load_turns

OUTPUT = ROOT / 'experiments' / 'feature_diagnostics' / 'roi_benchmark'


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {OUTPUT}')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

    import torch

    from train.encoders import save_cache

    from .benchmark import benchmark, select_turns, write_report, write_selection

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the CPU/CUDA benchmark')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    train = load_turns('train')
    turns = select_turns(train)
    ids = [t.turn_id for t in turns]
    print(f'Selected {len(turns)} unique Train turns', flush=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent, prefix='.probe-01-') as temp:
        out = Path(temp)
        caches, stats = benchmark(train, turns)
        sizes = {}
        for modality, rows in caches.items():
            path = save_cache(out, modality, ids, rows)
            size = path.stat().st_size
            sizes[path.name] = dict(actual=size, projected_1358=round(size * 1358 / len(turns)))
        stats['cache_bytes'] = sizes
        write_selection(out / 'selection.csv', turns, caches['video'])
        (out / 'benchmark.json').write_text(
            json.dumps(stats, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        write_report(out / 'report.md', stats)
        out.rename(OUTPUT)
    print(f'Six outputs written to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
