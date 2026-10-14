import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

m=json.loads(Path('results/metrics.json').read_text())['evaluations']
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
subsets=['test','gc_shift','length_shift'];x=np.arange(3)
for i,(name,label,color) in enumerate([('dinucleotide_boosting','Dinucleotide trees','#ad661b'),('transformer_ensemble','Transformer ensemble','#216e91')]):
    values=[m[f'{name}/{s}'] for s in subsets]
    axes[0].bar(x+(i-.5)*.35,[v['energy_rmse_kcal_mol'] for v in values],.35,label=label,color=color)
    axes[1].plot(x,[v['family_coverage'] for v in values],'o-',label=label,color=color)
axes[1].axhline(.9,ls='--',color='#444',label='Nominal 90%')
for ax in axes:ax.set_xticks(x,['Held-out families','GC shift','Length shift']);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Energy RMSE (kcal/mol)');axes[1].set_ylabel('All six relatives covered');axes[1].set_ylim(.5,1)
fig.legend(*axes[1].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False)
fig.savefig('results/benchmark.png',dpi=180);fig.savefig('results/benchmark.svg')
