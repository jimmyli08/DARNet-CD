# DARNet-CD: A Densely Attentive Refinement Network for Change Detection based on Very-High-Resolution Bi-Temporal Remote Sensing Images

This repo is the official implementation for DARNet proposed in the journal article "A Densely Attentive Refinement Network for Change Detection based on Very-High-Resolution Bi-Temporal Remote Sensing Images" accepted by IEEE Transactions on Geoscience and Remote sensing. More details about this work are described in the paper (https://ieeexplore.ieee.org/document/9734050). 

![DARNet](./images/Architecture.jpg)

## Requirements

- Python 3.7
- PyTorch 1.7.0
- Torchvision 0.8.0



## Datasets

### CDD

 [Change detection in remote sensing images using conditional adversarial networks](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2/565/2018/isprs-archives-XLII-2-565-2018.pdf)

### LEVIR-CD

 [A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://www.mdpi.com/2072-4292/12/10/1662)

### SYSUCD

 [A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection](https://ieeexplore.ieee.org/document/9467555/)

## Basic Usage

Prepare the training/validation/testing datasets as the examples in `data` directory.

### Train

`python train.py`

### Evaluate

`python evaluate.py`

### Inference

`python inference.py`

### Pretrained weights

The pretrained models can be downloaded soon.



## Citation

If you find this code useful and utilize it in your own research, please consider citing our article with the following bibtex:

```
@ARTICLE{9734050,
  author={Li, Ziming and Yan, Chenxi and Sun, Ying and Xin, Qinchuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Densely Attentive Refinement Network for Change Detection based on Very-High-Resolution Bi-Temporal Remote Sensing Images}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2022.3159544}}
```

## Contact

If you have any question about this code, please contact **Ziming Li**: lizm9@mail2.sysu.edu.cn



