import os

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
import subprocess
import sys

# Use the active Python environment, or set the Python path of your server environment.
python_path = sys.executable
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')


Deng = {'dropout': 0.4, 'rels': 65, 'batch_size': 1024, 'n_epochs': 400,
        'dataset': 'Deng', 'pair_method': 'avg', 'model': 'Model',  'fold': 0}

deepDDI = {'dropout': 0.4, 'rels': 86, 'batch_size': 1024, 'n_epochs': 200,
           'dataset': 'deepDDI', 'pair_method': 'cat', 'model': 'Model', 'fold': 0}

seed = 42
gpu = 0

datasets = [Deng,deepDDI]
folds = [0,1,2,3,4]


os.makedirs(os.path.join(os.path.dirname(script_path), 'result'), exist_ok=True)
for config in datasets:
    for fold in folds:
        cmd = [python_path, script_path, '--rels', str(config['rels']), '--n_epochs',
            str(config['n_epochs']), '--batch_size', str(config['batch_size']),
            '--dataset', config['dataset'], '--pair_method', config['pair_method'], '--model',config['model'],
            '--gpu_id', str(gpu), '--seed', str(seed), '--fold', str(fold),
            '--filename', f"result/{config['dataset']}_five_fold.csv"]
        subprocess.run(cmd, cwd=os.path.dirname(script_path), check=True)


