
Official implementation of our paper:

> [**Exploring the Boundaries of Semi-Supervised Facial Expression Recognition using In-Distribution, Out-of-Distribution, and Unconstrained Data**](https://arxiv.org/abs/2306.01195) <br>
> Shuvendu Roy, Ali Etemad<br>
> In IEEE Transactions on Affective Computing, 2024

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.01229)


<p align="center">
  <img src="https://github.com/ShuvenduRoy/SSL_FER/blob/main/figures/overview.png?raw=true" alt="drawing" width="600"/>
</p>

### Dataset
We used the following dataset 
1. [AffectNet](http://mohammadmahoor.com/affectnet/)
2. [FER-13](https://www.kaggle.com/datasets/msambare/fer2013)
3. [RAF-DB](http://www.whdeng.cn/RAF/model1.html)
4. [KDEF](https://www.kdef.se/)
5. [DDCF](https://lab.faceblind.org/k_dalrymple/ddcf)
6. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Once the dataset is downloaded use the scripts in `datasets/preprocessing` to preprocess the dataset.
The porcessed dataset structure should look like this:
```
dataset
├── train
│   ├── class_001
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   ├── class_002
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   └── ...
└── val
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```

### Run
Modify the config files in `config/` directory if needed.

```
python [ALGO_NAME].py --c [CONFIG_FILE]
```


### Acknowledgement
This codebase is build upon the following repositories:
- [TorchSSL](https://github.com/TorchSSL/TorchSSL)
- [SemiCLS](https://github.com/TencentYoutuResearch/Classification-SemiCLS)
- [CoMatch](https://github.com/salesforce/CoMatch)
<br>

We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.


 

