# KAGNNs -- Graph Neural Networks that use Kolmogorov Arnold Networks as their building blocks

This is the official repository for our paper [KAGNNs: Kolmogorov-Arnold Networks meet Graph Learning](https://arxiv.org/abs/2406.18380).

All experiments were run with python>=3.11.

The code is split by learning task (among node classification, graph classification, graph regression). Scripts are provided so experiments can be reproduced.

For our KAN-based models, we used the [efficient-kan implementation](https://github.com/Blealtan/efficient-kan).

Please cite our work if you use code from this repository:
```
@article{
      bresson2025kagnns,
      title={{KAGNN}s: Kolmogorov-Arnold Networks meet Graph Learning},
      author={Roman Bresson and Giannis Nikolentzos and George Panagopoulos and Michail Chatzianastasis and Jun Pang and Michalis Vazirgiannis},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2025},
      url={https://openreview.net/forum?id=03UB1MCAMr},
      note={}
}
```
