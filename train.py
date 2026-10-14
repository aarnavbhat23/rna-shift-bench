"""Train, calibrate on whole families, and evaluate prespecified shifts."""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from transformer import FoldSurrogate, encode


def features(sequence):
    alphabet = 'ACGU'
    return [len(sequence)] + [sequence.count(c)/len(sequence) for c in alphabet] + [
        sum(sequence[i:i+2] == a+b for i in range(len(sequence)-1))/(len(sequence)-1)
        for a in alphabet for b in alphabet]


def conformal_radius(errors, families, alpha=.1):
    maxima = [np.max(errors[families == family]) for family in np.unique(families)]
    rank = int(np.ceil((len(maxima)+1)*(1-alpha)))
    return float(np.sort(maxima)[rank-1]) if rank <= len(maxima) else float('inf')


def predict(model, tokens, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(batch.to(device)).cpu().numpy() for batch in tokens.split(128)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/oracle.csv'))
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    frame = pd.read_csv(args.data)
    tokens = encode(frame.sequence.tolist())
    y = np.c_[frame.energy_kcal_mol / frame.length, frame.paired_fraction].astype('float32')
    x = np.array([features(s) for s in frame.sequence], dtype='float32')
    masks = {s: (frame.split == s).to_numpy() for s in frame.split.unique()}
    train, validation = masks['train'], masks['validation']
    predictions, histories, checkpoints = [], [], []
    output = Path('results'); output.mkdir(exist_ok=True)
    started = time.time()
    for seed in [714, 715]:
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        model = FoldSurrogate().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.01)
        loader = DataLoader(TensorDataset(tokens[train], torch.from_numpy(y[train])), batch_size=128,
                            shuffle=True, generator=rng)
        best, stale, state, history = float('inf'), 0, None, []
        for epoch in range(args.epochs):
            model.train(); losses = []
            for batch, target in loader:
                optimizer.zero_grad()
                loss = ((model(batch.to(args.device))-target.to(args.device))**2).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); losses.append(float(loss.detach().cpu()))
            val = float(np.mean((predict(model, tokens[validation], args.device)-y[validation])**2))
            history.append({'epoch': epoch+1, 'training_mse': float(np.mean(losses)), 'validation_mse': val})
            print(seed, epoch+1, val, flush=True)
            if val < best:
                best, stale, state = val, 0, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
            if stale == 5:
                break
        model.load_state_dict(state)
        torch.save(state, output/f'transformer-{seed}.pt')
        predictions.append(predict(model, tokens, args.device)); histories.append(history)
        checkpoints.append(int(np.argmin([h['validation_mse'] for h in history]))+1)
    transformer = np.mean(predictions, axis=0)
    baseline = np.column_stack([HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, random_state=714)
                               .fit(x[train], y[train, j]).predict(x) for j in range(2)])
    report = {'records': len(frame), 'families': frame.family.nunique(), 'device': args.device,
              'parameters_per_transformer': sum(p.numel() for p in model.parameters()),
              'selected_epochs': checkpoints, 'elapsed_seconds': time.time()-started,
              'calibration': '90% parent-family simultaneous split conformal; exchangeability required', 'evaluations': {}}
    records = []
    for name, prediction in [('dinucleotide_boosting', baseline), ('transformer_ensemble', transformer)]:
        errors = np.abs(prediction[:, 0]-y[:, 0])*frame.length.to_numpy()
        radius = conformal_radius(errors[masks['calibration']], frame.family.to_numpy()[masks['calibration']])
        for subset in ['test', 'gc_shift', 'length_shift']:
            mask = masks[subset]
            coverage = errors[mask] <= radius
            families = frame.family.to_numpy()[mask]
            metrics = {'rows': int(mask.sum()), 'energy_rmse_kcal_mol': float(root_mean_squared_error(
                frame.energy_kcal_mol[mask], prediction[mask, 0]*frame.length[mask])),
                'paired_fraction_mae': float(mean_absolute_error(y[mask, 1], prediction[mask, 1])),
                'row_coverage': float(coverage.mean()), 'family_coverage': float(np.mean([
                    coverage[families == f].all() for f in np.unique(families)])), 'interval_half_width_kcal_mol': radius}
            report['evaluations'][f'{name}/{subset}'] = metrics
            for i in np.flatnonzero(mask):
                records.append({'id': frame.id[i], 'family': int(frame.family[i]), 'split': subset, 'model': name,
                                'observed_energy': float(frame.energy_kcal_mol[i]),
                                'predicted_energy': float(prediction[i, 0]*frame.length[i]), 'half_width': radius})
    pd.DataFrame(records).to_csv(output/'predictions.csv', index=False)
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'history.json').write_text(json.dumps(histories, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
