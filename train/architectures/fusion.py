from dataclasses import dataclass

import torch
from torch import nn

INPUT_DIMS = {'text': 384, 'audio': 2304, 'video': 162}


@dataclass(frozen=True)
class ModelConfig:
    modalities: tuple[str, ...]
    conditioning: str
    audio_projection: str = 'original'
    dropout: float = 0.3
    hidden_dim: int = 128
    role_dim: int = 16

    def __post_init__(self):
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError('Expected distinct, nonempty modalities')
        if not set(self.modalities) <= INPUT_DIMS.keys() or self.conditioning not in ('film', 'concat', 'none'):
            raise ValueError('Unsupported modality or role conditioning')
        if self.audio_projection not in ('original', 'shared') or (self.audio_projection == 'shared' and 'audio' not in self.modalities):
            raise ValueError('Invalid audio projection')


def projection(width, spec):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, spec.hidden_dim), nn.GELU(), nn.Dropout(spec.dropout))


def block(width, spec):
    return nn.Sequential(nn.Linear(width, spec.hidden_dim), nn.LayerNorm(spec.hidden_dim), nn.GELU(), nn.Dropout(spec.dropout))


class FusionModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        audio = None
        if spec.audio_projection == 'shared':
            audio = nn.Sequential(nn.Unflatten(1, (3, 768)), nn.LayerNorm(768), nn.Linear(768, 64),
                                  nn.Flatten(1), projection(192, spec))
        self.projections = nn.ModuleDict({m: audio if m == 'audio' and audio is not None else projection(INPUT_DIMS[m], spec)
                                          for m in spec.modalities})
        self.layer_logits = nn.Parameter(torch.zeros(13)) if 'audio' in spec.modalities else None
        self.fusion = block(len(spec.modalities) * spec.hidden_dim, spec)
        self.role_embedding = nn.Embedding(2, spec.role_dim) if spec.conditioning != 'none' else None
        self.film = None
        if spec.conditioning == 'film':
            self.film = nn.Linear(spec.role_dim, 2 * spec.hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.role_block = block(spec.hidden_dim + (spec.role_dim if spec.conditioning == 'concat' else 0), spec)
        self.expr_head = nn.Linear(spec.hidden_dim, 8)
        self.int_head = nn.Linear(spec.hidden_dim, 2)

    def diagnostics(self):
        if self.layer_logits is None:
            return {'audio_branch_parameters': 0}
        audio = (self.layer_logits, *self.projections['audio'].parameters())
        return {'audio_branch_parameters': sum(p.numel() for p in audio if p.requires_grad),
                'wavlm_layer_weights': self.layer_logits.detach().softmax(0).cpu().tolist()}

    def forward(self, inputs):
        projected = []
        for name, project in self.projections.items():
            x = inputs[name]
            if name == 'audio':
                x = torch.einsum('l,bslh->bsh', self.layer_logits.softmax(0), x.transpose(1, 2).contiguous()).flatten(1)
            projected.append(project(x))
        z = self.fusion(torch.cat(projected, dim=1))
        if self.role_embedding is not None:
            embedding = self.role_embedding(inputs['role'])
            if self.film is None:
                z = torch.cat((z, embedding), dim=1)
            else:
                delta_gamma, beta = self.film(embedding).chunk(2, dim=1)
                z = (1 + delta_gamma) * z + beta
        z = self.role_block(z)
        return {'expression_logits': self.expr_head(z), 'intensity_logits': self.int_head(z)}


def build_model(config):
    return FusionModel(config)
