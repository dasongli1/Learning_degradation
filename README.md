# Learning Degradation Representations for Image Deblurring

This is an official implementation of “Learning Degradation Representations for Image Deblurring” with PyTorch, accepted by ECCV 2022.

<center><img src="figures/network.png "width="60%"></center>

**Abstract**: In various learning-based image restoration tasks, such as image denoising and image super-resolution, the degradation representations were widely used to model the degradation process and handle complicated degradation patterns.
However, they are less explored in learning-based image deblurring as blur kernel estimation cannot perform well in real-world challenging cases. We argue that it is particularly necessary for image deblurring to model degradation representations since blurry patterns typically show much larger variations than noisy patterns or high-frequency textures.
In this paper, we propose a framework to learn spatially adaptive degradation representations of blurry images. A novel joint image reblurring and deblurring learning process is presented to improve the expressiveness of degradation representations. 
To make learned degradation representations effective in reblurring and deblurring, we propose a Multi-Scale Degradation Injection Network (MSDI-Net) to integrate them into the neural networks. With the integration, MSDI-Net can handle various and complicated blurry patterns adaptively. 
Experiments on the GoPro and RealBlur datasets demonstrate that our proposed deblurring framework with the learned degradation representations outperforms state-of-the-art methods with appealing improvements.

**Keywords**: Image Deblurring, Degradation Representations

## Performance
<center><img src="figures/results.png "width="80%"></center>



## Get Started

### Installation
```python
python 3.8.5
pytorch 1.8.0
cuda 11.3
```

```
git clone https://github.com/dasongli1/Learning_degradation.git
cd Learning_degradation
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Image Deblurring
* download the [train](https://drive.google.com/drive/folders/1AsgIP9_X0bg0olu2-1N6karm2x15cJWE) set in ./datasets/GoPro/train and [test](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf) set in ./datasets/GoPro/test (refer to [MPRNet](https://github.com/swz30/MPRNet)) 
  * it should be like:
  
    ```bash
    ./datasets/
    ./datasets/GoPro/
    ./datasets/GoPro/train/
    ./datasets/GoPro/train/input/
    ./datasets/GoPro/train/target/
    ./datasets/GoPro/test/
    ./datasets/GoPro/test/input/
    ./datasets/GoPro/test/target/
    ```
  
  * ```python gopro.py```
  
### Training:
* Download the model of [net_Encoder](https://drive.google.com/file/d/1T7Pf065mt9bm801bVOAjmOA8zYKoMz2m/view?usp=sharing) and [prior_upsampling](https://drive.google.com/file/d/168YGqQ9rBSGavOb-TlyQkaLGEKSkpBqu/view?usp=sharing) to ./checkpoints/

* training script:
```
python -m torch.distributed.launch --nproc_per_node=8 basicsr/train.py -opt MSDINet-Train.yml --launcher pytorch
```
 
### Testing: 
* eval: We provide the pre-trained model for evaluation.
* Please download the model [pretrained model](https://drive.google.com/file/d/1HB06DPJ2bydHhjjuxmVGrQ7F63dbaKXL/view?usp=sharing) to ./checkpoints/msdi_net.pth
* ```python basicsr/test.py -opt MSDINet-Test.yml ```
