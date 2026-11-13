# AGAT-DDI

AGAT-DDI predicts drug-drug interaction (DDI) events by integrating chemical, target, enzyme, and pathway attributes with graph representations. The model combines an attribute autoencoder, relational and similarity-augmented graph encoders, attention-based fusion, and a DDI event classifier.

## Installation

The server environment uses Python 3.8.20, PyTorch 2.1.0, PyTorch Geometric 2.5.3, and CUDA 12.1 on Linux.

```bash
conda env create -f environment_server.yaml
conda activate AGAT-DDI
```

GPU training requires an NVIDIA driver compatible with CUDA 12.1.

## Data

| Dataset | Drugs | DDI records | Event classes |
|---|---:|---:|---:|
| Deng | 572 | 37,264 | 65 |
| deepDDI | 1,704 | 191,870 | 86 |

Each dataset directory contains four precomputed similarity matrices and five predefined train/test splits:

```text
Deng/ or deepDDI/
|-- drug_list.csv
|-- chem_Jacarrd_sim.csv
|-- target_Jacarrd_sim.csv
|-- enzyme_Jacarrd_sim.csv
|-- pathway_Jacarrd_sim.csv
`-- 0/ ... 4/
    |-- train.csv
    `-- test.csv
```

The split files contain `Drug1,Drug2,Label`. Drug indices and labels are zero-based. Similarity matrices are numeric CSV files without headers or index columns, ordered consistently with `drug_list.csv`. The supplied splits use approximately 80% training and 20% test data and are loaded directly by the code.

## Usage

Run commands from the repository root.

To run all five folds on both datasets:

```bash
python Run.py
```

Dataset settings, folds, training seed, and GPU index can be configured in `Run.py`. The default settings use 400 epochs and average pair representations for Deng, and 200 epochs and concatenated pair representations for deepDDI.

To run a single fold:

```bash
# Deng: 65 event classes
python main.py --dataset Deng --rels 65 --fold 0 --pair_method avg --n_epochs 400 --filename Deng_fold0.csv

# deepDDI: 86 event classes
python main.py --dataset deepDDI --rels 86 --fold 0 --pair_method cat --n_epochs 200 --filename deepDDI_fold0.csv
```

Use `--gpu_id`, `--seed`, `--batch_size`, and `--lr` to customize training. Keep `--rels` consistent with the selected dataset. Run `python main.py --help` for the complete argument list.

## Results

`Run.py` appends fold-level results to `result/Deng_five_fold.csv` and `result/deepDDI_five_fold.csv`. Reported metrics are accuracy, micro-AUPR, micro-AUROC, macro-F1, macro-precision, and macro-recall. Repeated runs append additional rows.

## Main files

- `main.py`: data loading, graph construction, training, and evaluation.
- `model.py`: the full model and ablation variants.
- `Run.py`: five-fold experiment launcher.
- `calculate_attribute_jaccard.py`: computes target, enzyme, and pathway Jaccard similarity matrices in drug-list order.
- `extract_drugbank_features.py`: extracts target, enzyme, and pathway attributes from DrugBank XML for the selected drugs.
- `five_fold_result/`: included per-class results and an averaging helper.