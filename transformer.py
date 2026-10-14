"""A small, from-scratch sequence encoder; no pretrained weights."""
import torch
from torch import nn


def encode(sequences):
    result = torch.zeros((len(sequences), max(map(len, sequences))), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        if not sequence or set(sequence) - set('ACGU'):
            raise ValueError('Expected a nonempty canonical RNA sequence')
        result[i, :len(sequence)] = torch.tensor(['ACGU'.index(c)+1 for c in sequence])
    return result


class FoldSurrogate(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.tokens = nn.Embedding(5, width, padding_idx=0)
        self.positions = nn.Embedding(256, width)
        layer = nn.TransformerEncoderLayer(width, 4, width*2, .1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(width+2, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, tokens):
        mask = tokens != 0
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.tokens(tokens) + self.positions(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        gc = ((tokens == 2) | (tokens == 3)).sum(1) / mask.sum(1)
        covariates = torch.stack([gc, mask.sum(1)/100], 1)
        return self.head(torch.cat([pooled, covariates], 1))
