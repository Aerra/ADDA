# Pytorch Adversarial Domain Adaptation
A collection of implementations of adversarial unsupervised domain adaptation algorithms.

**Paper**: Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)  
**Link**: [https://arxiv.org/abs/1702.05464](https://arxiv.org/abs/1702.05464)  
**Description**: Adapts the weights of a classifier pretrained on source data to produce similar features on the target data.  
**Implementation**: [adda.py](https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/adda.py)

1. Train a model on the source dataset with
```
$ python train_source.py
```
2. Choose an algorithm and pass it the pretrained network, for example:
```
$ python adda.py trained_models/source.pt
```
