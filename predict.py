"""Run the two released surrogate checkpoints on one RNA sequence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformer import FoldSurrogate,encode


def main():
    parser = argparse.ArgumentParser();parser.add_argument('sequence');args=parser.parse_args()
    if len(args.sequence)>255: parser.error('Positional embedding limit is 255 nucleotides')
    estimates = []
    for seed in [714,715]:
        model = FoldSurrogate().eval()
        model.load_state_dict(torch.load(Path('results')/f'transformer-{seed}.pt',weights_only=True,map_location='cpu'))
        with torch.no_grad(): estimates.append(model(encode([args.sequence])).numpy()[0])
    prediction = np.mean(estimates,0)
    print(json.dumps({'energy_kcal_mol':float(prediction[0]*len(args.sequence)),
                      'paired_fraction':float(prediction[1]),'note':'Oracle surrogate; not an experimental measurement. Fractions are unconstrained regression outputs.'},indent=2))


if __name__ == '__main__':main()
