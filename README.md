# AD-VQ: Annealed Dual Vector Quantization for Molecular OOD Generalization

This is the official implementation of the paper **"Towards Out-of-Distribution Generalizable Molecular Representation: A Dual Vector Quantization Framework with Annealed Noise"**.

## 1. Requirements

### Environment
The code is implemented based on **PyTorch** and **PyTorch Geometric (PyG)**. 
This repo is also depended on `GOOD` and `DrugOOD`, please follow the installation methods provided for each package:

* GOOD (Version 1.1.1)
  * Repository: https://github.com/divelab/GOOD/
  * Installation: Please follow the instructions provided in the repository to install.
* DrugOOD (Version 0.0.1)
  * Repository: https://github.com/tencent-ailab/DrugOOD
  * Installation: Please follow the instructions provided in the repository to install. 

```bash
conda create -n advq python=3.9
conda activate advq
pip install torch==1.12.1+cu113 -f [https://download.pytorch.org/whl/cu113/torch_stable.html](https://download.pytorch.org/whl/cu113/torch_stable.html)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install vector-quantize-pytorch
