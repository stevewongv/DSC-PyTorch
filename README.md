# DSC-PyTorch
This is a PyTorch implementation of ["Direction-Aware Spatial Context Features for Shadow Detection, CVPR'18"](https://arxiv.org/abs/1712.04142) and detection part of ["Direction-Aware Spatial Context Features for Shadow Detection and Removal, T-PAMI'2019"](https://arxiv.org/abs/1805.04635) based on [Xiaowei](https://xw-hu.github.io)'s [DSC (Caffe)](https://github.com/xw-hu/DSC) written by Tianyu Wang.

The Spacial IRNN is implemented by using CUDA 9.0. The backbone is ResNeXt101 pre-trained on ImageNet and the implementation of loss is from [Quanlong Zheng](https://quanlzheng.github.io). 

## Results
We use two GTX 1080Ti to train the DSC on SBU dataset.

### SBU
| Methods | BER | Accuracy |
| --- | --- | --- |
| DSC (Caffe) | 5.59 |**0.97** |
| DSC (Our) | **5.43** | 0.96 |

You can download the pre-trained model from [Google Drive](https://drive.google.com/file/d/17VfUOu5xwHHc3M05N0oCjF2FGGirw7gt/view?usp=sharing) and put it into `SBU_model` folder.
You can download the ResNeXt101 model from [Google Drive](https://drive.google.com/open?id=1EDUcaGNiakWO9Xvk9kWgkkcnTYZ6VQoT) and put it in main folder.

## Requirements
* PyTorch == 0.4.1 (1.0.x may not work for training)
* Cupy ([Installation Guide](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy))
* TensorBoardX
* Python3.6
* progressbar2
* scikit-image
* pydensecrf

## Train/Test
1. **Clone this repository**

```bash
git clone https://github.com/stevewongv/DSC-PyTorch.git
```
2. **Train**

```bash
python3 main.py -a train
```
3. **Test**

```bash
python3 main.py -a test
```

## TODO List
* [x] ResNext101 Backbone
* [x] Test on SBU Test Set
* [ ] VGG19 Backbone
* [ ] Test on ISTD Test Set
* [ ] Test on UCF Test Set
* [ ] ...