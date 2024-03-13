# SGLFormer: Spiking Global-Local-Fusion Transformer with High Performance, [This is Link]( https://doi.org/10.3389/fnins.2024.1371290)
Our models achieve SOTA performance on several datasets (eg. 83.73 % on ImageNet, 96.76 % on CIFAR10, 82.26 % on CIFAR100, 82.9% on CIFAR10-DVS) in directly trained SNNs in 2024/03.

## Reference
If you find this repo useful, please consider citing:
```
@ARTICLE{10.3389/fnins.2024.1371290,
  AUTHOR={Zhang, Han  and Zhou, Chenlin  and Yu, Liutao  and Huang, Liwei  and Ma, Zhengyu  and Fan, Xiaopeng  and Zhou, Huihui  and Tian, Yonghong },
  TITLE={SGLFormer: Spiking Global-Local-Fusion Transformer with high performance},
  JOURNAL={Frontiers in Neuroscience},
  VOLUME={18},
  YEAR={2024},
  URL={https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1371290},
  DOI={10.3389/fnins.2024.1371290},
  ISSN={1662-453X}
}
```
Our codes are based on ...

The code will be made public after the paper is published. 


## Results on ImageNet-1K

| Model              | Resolution| T     |  Param.     |Top-1 Acc|
| :---               | :---:     | :---: | :---:       |:---:    |
| SGLFormer-8-384    | 224x224   | 4     |  16.25M     |72.73    |
| SGLFormer-8-512    | 224x224   | 4     |  28.67M     |75.61    |
| SGLFormer-8-512*   | 224x224   | 4     |  28.67M     |77.34    |
| SGLFormer-8-768*   | 224x224   | 4     |  64.02M     |77.64    |


## Results on CIFAR10/CIFAR100

| Model               | T      |  Param.     | CIFAR10 Top-1 Acc |CIFAR100 Top-1 Acc|
| :---                | :---:  | :---:       |  :---:            |:---:  |
| SGLFormer-4-384     | 4      |  8.85/8.88M | 96.76             |82.26  |


## Results on CIFAR10-DVS/DVS128

| Model            |  Dataset    | T      |  Param.     |   Top-1 Acc |
| :---             | :---:       | :---:  | :---:       |:---:        |
| SGLFormer-3-256  | CIFAR10 DVS | 10     |  2.48M      | 82.9        |
| SGLFormer-3-256  | CIFAR10 DVS | 16     |  2.58M      | 82.6        |
| SGLFormer-3-256  | DVS 128     | 10     |  2.08M      | 97.2        |
| SGLFormer-3-256  | DVS 128     | 16     |  2.17M      | 98.6        |


## Requirements
For imagenet
timm==0.6.12; cupy==11.3.0; torch==1.14.0+cu116; spikingjelly==0.0.0.0.12; pyyaml;
For others
timm==0.6.12; cupy==11.3.0; torch==1.14.0+cu116; spikingjelly==0.0.0.0.12; pyyaml;

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```







