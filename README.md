# SGLFormer: Spiking Global-Local-Fusion Transformer with high performance, [this is link]( https://doi.org/10.3389/fnins.2024.1371290)
Our models achieve SOTA performance on several datasets (eg. 83.73 % on ImageNet, 96.76 % on CIFAR10, 82.26 % on CIFAR100, 82.9% on CIFAR10-DVS) in directly trained SNNs in 2024/03.


## 
Our codes are based on [QKFormer](https://github.com/zhouchenlin2096/QKFormer) and [Spikformer](https://github.com/ZK-Zhou/spikformer).

##
The trained model for CIFAR100 and ImageNet is provided at [BauduYun](https://pan.baidu.com/s/1F5vsg9V4Vaj4jbTC103m3A?pwd=thd3).

## 
<p align="center">
<img src="https://github.com/ZhangHanN1/SGLFormer/tree/main/img/performance-energy-SGLFormer-QKFormer.jpg">
</p>

<p align="center">
<img src="https://github.com/ZhangHanN1/SGLFormer/tree/main/img/SGLFormer.jpg">
</p>

## Results on ImageNet-1K

| Model              | Resolution| T     |  Param.     |Top-1 Acc|
| :---               | :---:     | :---: | :---:       |:---:    |
| Swin Transformer   | 224x224   | -     |  88M        |83.5     |
| SGLFormer-8-384    | 224x224   | 4     |  16.25M     |79.44    |
| SGLFormer-8-512    | 224x224   | 4     |  28.67M     |82.28    |
| SGLFormer-8-512*   | 224x224   | 4     |  28.67M     |81.93    |
| SGLFormer-8-768*   | 224x224   | 4     |  64.02M     |83.73    |


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
timm==0.3.2 for imagenet, timm==0.6.12 for others; cupy==9.6.0; torch==1.10.0; cuda==11.3.1; cudnn==8.2.1; spikingjelly==0.0.0.0.12; pyyaml==5.3.1;

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

## Train
### Training  on ImageNet
```
cd imagenet
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py --accum_iter 2 --batch_size 32 --blr 6e-4 --model sglformer_ima_8_512 --output_dir ./sglformer_ima_8_512
# sglformer_ima_8_512 is SGLFormer-8-512*
# sglformer_ima2_8_512 is SGLFormer-8-512
```

### Training  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

### Training  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar100
python train.py
```

### Training  on DVS128 Gesture
```
cd dvs128-gesture
python train.py --T=10 --lr=5e-3
```

### Training  on CIFAR10-DVS
```
cd cifar10-dvs
python train.py --T=10 --lr=5e-3
```


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





