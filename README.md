# DSC-PyTorch
This is a PyTorch implementation of ["Direction-Aware Spatial Context Features for Shadow Detection, CVPR'18"](https://arxiv.org/abs/1712.04142) and ["Direction-Aware Spatial Context Features for Shadow Detection and Removal, T-PAMI'19"](https://arxiv.org/abs/1805.04635) based on [Xiaowei](https://xw-hu.github.io)'s [DSC (Caffe)](https://github.com/xw-hu/DSC) written by Tianyu Wang.

The Spacial IRNN is implemented by using CUDA 11.x. The backbone is ResNeXt101 pre-trained on ImageNet and the implementation of loss is from [Quanlong Zheng](https://quanlzheng.github.io). 

## Results
We use two GTX 1080Ti to train the DSC on SBU dataset.

### SBU
| Methods | BER | Accuracy |
| --- | --- | --- |
| DSC (Caffe) | 5.59 |**0.97** |
| DSC (Our) | **5.19** | 0.95 |

**Pre-trained model is available. You can download from [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155152065_link_cuhk_edu_hk/EcO20MV0kSVKkEbXO2NVIWMB6jewfk_lJK4SJjDvHcB6Ag?e=6P2h0m) and put it into `SBU_model` folder.**

* You can download the ResNeXt101 model from [Google Drive](https://drive.google.com/open?id=1EDUcaGNiakWO9Xvk9kWgkkcnTYZ6VQoT) and put it in main folder.

## Requirements
* PyTorch == 1.8.1 (training and testing)
* Cupy ([Installation Guide](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy))
* TensorBoardX
* Python
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
python3 main.py -a train    # For Shadow Detection
python3 main_sr.py -a train # For Shadow Removal
```
3. **Test**

```bash
python3 main.py -a test    # For Shadow Detection
python3 main_sr.py -a test # For Shadow Removal
```

## Citations

```
@InProceedings{Hu_2018_CVPR,      
    author = {Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},      
    title = {Direction-Aware Spatial Context Features for Shadow Detection},      
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},      
    pages={7454--7462},        
    year = {2018}
}

@article{hu2019direction,   
    author = {Hu, Xiaowei and Fu, Chi-Wing and Zhu, Lei and Qin, Jing and Heng, Pheng-Ann},    
    title = {Direction-Aware Spatial Context Features for Shadow Detection and Removal},    
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},    
    year  = {2019},          
    note={to appear}                  
}

Modified DSC module is used in SPANet:

@InProceedings{Wang_2019_CVPR,
  author = {Wang, Tianyu and Yang, Xin and Xu, Ke and Chen, Shaozhe and Zhang, Qiang and Lau, Rynson W.H.},
  title = {Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

## TODO List
* [x] ResNext101 Backbone
* [x] Test on SBU Test Set
* [ ] VGG19 Backbone
* [ ] Test on ISTD Test Set
* [ ] Test on UCF Test Set
* [ ] ...
