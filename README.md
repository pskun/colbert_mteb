<h1 align="center">Massive Text Embedding Benchmark</h1>

## Installation

```bash
conda create -n cmteb python=3.10
conda activate cmteb
pip install -r requirements.txt
```

## Usage

```shell
sbatch run_mteb.sh
```

For Chinese tasks, you can refer to [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB).

## Citation

If you find MTEB useful, feel free to cite our publication [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316):

```bibtex
@article{muennighoff2022mteb,
  doi = {10.48550/ARXIV.2210.07316},
  url = {https://arxiv.org/abs/2210.07316},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},  
  year = {2022}
}
```
