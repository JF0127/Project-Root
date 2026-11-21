import hashlib
import json
import subprocess
from contextlib import contextmanager

from .config import write_json
from .data import CACHE, CATALOG, MANIFEST, ROOT
from .metrics import scalar_fields


def digest_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def provenance():
    cache = ROOT / CACHE
    inputs = [ROOT / CATALOG, ROOT / MANIFEST]
    catalog = json.loads(inputs[0].read_text(encoding='utf-8'))
    inputs.append(ROOT / catalog['splits_and_statistics']['membership'])
    inputs.extend(ROOT / catalog['usage_rules_and_limitations'][key]['path'] for key in ('training', 'validation'))
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
            'code_sha256': digest_config({str(p.relative_to(ROOT)): file_digest(p) for p in (ROOT / 'train').rglob('*.py')}),
            'dataset_sha256': digest_config({str(p.relative_to(ROOT)): file_digest(p) for p in inputs}),
            'features_sha256': digest_config({str(p.relative_to(cache)): file_digest(p) for split in ('train', 'val')
                                              for p in (cache / split).glob('*_features.npz')}),
            'uv_lock_sha256': file_digest(ROOT / 'uv.lock')}


class Tracker:
    def __init__(self, run, artifact):
        self.run = run
        self.artifact = artifact

    def epoch(self, epoch, curves):
        if self.run is not None:
            self.run.log({'epoch': epoch, **curves}, step=epoch)

    def complete(self, result, checkpoint, normalization):
        if self.run is None:
            return
        step = result['stopped_epoch']
        scores = {f'final/{k.replace(".", "/")}': v for k, v in scalar_fields(result).items() if v is not None}
        self.run.log(scores, step=step)
        for subset in ('val_strict', 'val_all'):
            matrix = result[subset]['expression']['confusion_matrix']
            self.run.log_text(key=f'final/{subset}/expression/confusion_matrix', data=json.dumps(matrix), step=step)
        if self.artifact:
            config_path = checkpoint.with_name('inference_config.json')
            write_json(config_path, {'model': self.run.config['model'], 'video_zscore': normalization})
            for path in (checkpoint, config_path):
                self.run.save(path, base_path=checkpoint.parent, policy='now')


@contextmanager
def track_seed(directory, study, name, seed, config, mode='offline', artifact=False):
    if mode not in ('online', 'offline', 'disabled'):
        raise ValueError('Unknown SwanLab mode')
    if mode == 'disabled':
        if artifact:
            raise ValueError('Artifacts require offline or online tracking')
        yield Tracker(None, False)
        return
    import swanlab

    settings = swanlab.Settings(interactive=False, terminal=swanlab.Settings.Terminal(proxy_type='none'),
                               probe=swanlab.Settings.Probe(hardware=False, runtime=False, requirements=False,
                                                            conda=False, git=False, swanlab=False, monitor=False))
    run = swanlab.init(project='robot-expression', group=study.replace('_', '-'), name=f'{name}-seed-{seed}',
                       log_dir=str(directory / 'swanlog'), mode=mode, config=config, settings=settings)
    try:
        yield Tracker(run, artifact)
    except KeyboardInterrupt:
        run.finish(state='aborted')
        raise
    except BaseException as exc:
        run.finish(state='crashed', error=type(exc).__name__)
        raise
    else:
        run.finish()
