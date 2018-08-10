# pytorch-SBADA-GAN

This is an unofficial pytorch implementation of algorithms for domain adaptation.

## Requirements
- Python 3.5+
- PyTorch 0.4
- TorchVision
- TensorboardX
- batchup
- click

## Usage

These examples are for the MNIST to USPS experiment.

### Train `Source Only` Model
```
CUDA_VISIBLE_DEVICES=$(nvidia-empty) python train_classifier.py --exp mnist_usps --train_type unsup
```

### Train `Target Only` Model
```
CUDA_VISIBLE_DEVICES=$(nvidia-empty) python train_classifier.py --exp mnist_usps --train_type sup
```

### Train Model
```
UDA_VISIBLE_DEVICES=
$(nvidia-empty) python test_classifier.py --exp mnist_usps --snapshot <snapshot_dir>
```
