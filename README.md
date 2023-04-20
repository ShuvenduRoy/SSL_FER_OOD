
# FER_SSL


<p align="center">
  <img src="https://github.com/ShuvenduRoy/SSL_FER/blob/main/figures/overview.png?raw=true" alt="drawing" width="600"/>
</p>

### Dataset
We used the following dataset 
1. [AffectNet](http://mohammadmahoor.com/affectnet/)
2. [FER-13](https://www.kaggle.com/datasets/msambare/fer2013)
3. [RAF-DB](http://www.whdeng.cn/RAF/model1.html)

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


 

