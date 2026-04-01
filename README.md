# Lit2Vec TL;DR – Training Code

[![Hugging Face Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset)
[![Hugging Face Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/Bocklitz-Lab/lit2vec-tldr-bart)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

This repository contains the **training pipeline** for fine-tuning a lightweight DistilBART model on the  
[Lit2Vec TL;DR Chemistry Dataset](https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset).  

The goal: generate **legally compliant, schema-consistent TL;DR-style summaries** of chemistry abstracts, suitable for downstream use in semantic search, knowledge graphs, and scientific indexing.  

---

## 🚀 Quickstart

### 1. Clone repo & install requirements
```bash
git clone https://github.com/Bocklitz-Lab/lit2vec-tldr-bart.git
cd lit2vec-tldr-bart

# with conda
conda create -n lit2vec python=3.10 -y
conda activate lit2vec
pip install -r requirements.txt

# or with pip
pip install -r requirements.txt
````

### 2. (Optional) Enable Weights & Biases logging

```bash
wandb login
```

### 3. Download dataset from Hugging Face

The training script pulls the dataset automatically:

```python
from datasets import load_dataset
ds = load_dataset("Bocklitz-Lab/lit2vec-tldr-bart-dataset")
```

### 4. Run training

```bash
python src/lit2vec_tldr/train.py --config configs/train_distilbart.yaml
```

Checkpoints will be saved in `saved_data/checkpoints/`,
the final model in `saved_data/final_model/`.

---

## 📂 Repo Structure

```
lit2vec-tldr-bart/
├── README.md               # this file
├── LICENSE
├── CITATION.cff            # citation metadata
├── requirements.txt
├── configs/
│   ├── train_distilbart.yaml   # training hyperparameters
│   └── eval.yaml               # evaluation configs
├── src/
│   └── lit2vec_tldr/
│       ├── train.py            # main training loop
│       ├── preprocess.py       # tokenization & preprocessing
│       ├── metrics.py          # ROUGE & helpers
│       └── utils.py            # seeds, logging, etc.
├── scripts/
│   ├── run_train.sh
│   └── export_model.sh
├── saved_data/                 # (ignored by git)
│   ├── cached_tokenized_data/
│   ├── checkpoints/
│   └── final_model/
└── evaluation/
    ├── test_metrics.json       # final evaluation results
    └── plots/
```

---

## 📊 Reproducing Paper Results

We provide:

* **Dataset** → [lit2vec-tldr-bart-dataset](https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset) (19,992 abstracts)
* **Fine-tuned Model** → [lit2vec-tldr-bart](https://huggingface.co/Bocklitz-Lab/lit2vec-tldr-bart)
* **Evaluation results** → stored in [`evaluation/test_metrics.json`](./evaluation/test_metrics.json)

Run a final test evaluation:

```bash
python src/lit2vec_tldr/train.py --config configs/eval.yaml --evaluate
```

---

## 📖 Dataset

* **Source**: 19,992 chemistry abstracts (CC BY licensed only)
* **Input**: `abstract` field
* **Target**: `summary` field (TL;DR-style, schema-consistent)
* **Splits**: `train` (17,992) / `validation` (999) / `test` (1,001)

👉 Full dataset card:
[https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset](https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset)

---

## 🧪 Evaluation Metrics

* **ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum**
* Sentence segmentation handled with `nltk`
* Final reported values are in `evaluation/test_metrics.json`

---

## 🛠️ Requirements

* Python ≥ 3.10
* [PyTorch](https://pytorch.org/) with CUDA
* Hugging Face `transformers`, `datasets`, `evaluate`
* `nltk`, `tqdm`, `wandb` (optional for logging)

Install via:

```bash
pip install -r requirements.txt
```

---

## 📜 Citation

If you use this code or dataset, please cite:

```bibtex
@misc{lit2vec_tldr_2025,
  author       = {Bocklitz Lab},
  title        = {Lit2Vec TL;DR: Legally Compliant Abstractive Summaries for Chemistry Abstracts},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/Bocklitz-Lab/lit2vec-tldr-bart-dataset}},
  note         = {Submitted to Nature Scientific Data}
}
```

---

## 🙌 Acknowledgements

* Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers)
* Dataset derived from [Semantic Scholar](https://www.semanticscholar.org/) metadata and CC-BY publishers
* Initial summaries generated with GPT-4o, distilled into DistilBART

---