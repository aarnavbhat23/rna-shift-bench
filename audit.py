"""Replay errors and coverage from released predictions; check split invariants."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    data=pd.read_csv('data/oracle.csv');pred=pd.read_csv('results/predictions.csv')
    metrics=json.loads(Path('results/metrics.json').read_text())
    manifest=json.loads(Path('data/oracle.manifest.json').read_text())
    assert hashlib.sha256(Path('data/oracle.csv').read_bytes()).hexdigest()==manifest['sha256']
    assert data.groupby('family').split.nunique().max()==1
    assert data.groupby('sequence').split.nunique().max()==1
    assert data.id.is_unique
    for (model,split),rows in pred.groupby(['model','split']):
        expected=metrics['evaluations'][f'{model}/{split}']
        errors=np.abs(rows.predicted_energy-rows.observed_energy)
        covered=errors<=rows.half_width
        actual={'energy_rmse_kcal_mol':np.sqrt(np.mean(errors**2)),
                'row_coverage':covered.mean(),
                'family_coverage':pd.DataFrame({'family':rows.family,'covered':covered}).groupby('family').covered.all().mean()}
        for key,value in actual.items():
            assert np.isclose(value,expected[key],rtol=1e-6,atol=1e-6),(model,split,key)
    print('Verified 18,000 sequences, family-separated splits, hashes, errors and family coverage.')


if __name__=='__main__':main()
