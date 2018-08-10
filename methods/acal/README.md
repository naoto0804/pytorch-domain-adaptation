# pytorch-ACAL

This is an unofficial pytorch implementation of a paper, Augmented Cyclic Adversarial Learning for Domain Adaptation [Hosseini-Asl+, arXiv2018].

Please note that this is an ongoing project and I cannot fully reproduce the results currently.


### Pretrain classifier
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --pretrain --exp svhn_mnist_rgb
```

## References
- [1]: E. Hosseini-Asl et al. "Augmented Cyclic Adversarial Learning for Domain Adaptation.", in arXiv, 2018.
