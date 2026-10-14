"""A deterministic mutation-family benchmark with ViennaRNA oracle labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import RNA

ALPHABET = 'ACGU'


def mutate(sequence, count, rng):
    letters = list(sequence)
    for position in rng.choice(len(letters), size=count, replace=False):
        letters[position] = rng.choice([c for c in ALPHABET if c != letters[position]])
    return ''.join(letters)


def oracle(sequence):
    structure, energy = RNA.fold(sequence)
    return float(energy), sum(c != '.' for c in structure) / len(sequence), structure


def generate(families=2400, seed=714):
    rng = np.random.default_rng(seed)
    family_order = rng.permutation(families)
    split = {int(f): ('train' if i < .7*families else 'validation' if i < .8*families
                         else 'calibration' if i < .9*families else 'test')
             for i, f in enumerate(family_order)}
    for family in range(families + 600):
        if family < families:
            length, gc, subset = int(rng.choice([48, 64, 80, 96])), float(rng.uniform(.3, .7)), split[family]
        elif family < families + 300:
            length, gc, subset = int(rng.choice([48, 64, 80, 96])), float(rng.choice([.1, .9])), 'gc_shift'
        else:
            length, gc, subset = 128, float(rng.uniform(.3, .7)), 'length_shift'
        parent = ''.join(rng.choice(list(ALPHABET), size=length, p=[(1-gc)/2, gc/2, gc/2, (1-gc)/2]))
        seen = set()
        for variant, changes in enumerate([0, 1, 2, 4, 8, 12]):
            sequence = parent if not changes else mutate(parent, changes, rng)
            if sequence in seen:
                continue
            seen.add(sequence)
            energy, paired, structure = oracle(sequence)
            yield {'id': f'{family}:{variant}', 'family': family, 'split': subset,
                   'sequence': sequence, 'length': length, 'mutations': changes,
                   'energy_kcal_mol': energy, 'paired_fraction': paired, 'structure': structure}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/oracle.csv'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        writer = None
        for i, row in enumerate(generate()):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            if i % 3000 == 0:
                print('oracle rows', i, flush=True)
    manifest = {'oracle': 'ViennaRNA', 'version': RNA.__version__, 'temperature_celsius': 37,
                'seed': 714, 'families': 3000, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'provenance': 'Computational labels; not experimental RNA measurements.',
                'split_rule': 'Mutation relatives share a parent-family split; 600 shifted families are never used for fitting.'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
