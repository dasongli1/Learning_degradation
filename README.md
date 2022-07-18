# Learning Degradation Representations for Image Deblurring

This is an official implementation of “Learning Degradation Representations for Image Deblurring” with PyTorch, accepted by ECCV 2022.

<center><img src="figures/network.png "width="80%"></center>

**Abstract**: In various learning-based image restoration tasks, such as image denoising and image super-resolution, the degradation representations were widely used to model the degradation process and handle complicated degradation patterns.
However, they are less explored in learning-based image deblurring as blur kernel estimation cannot perform well in real-world challenging cases. We argue that it is particularly necessary for image deblurring to model degradation representations since blurry patterns typically show much larger variations than noisy patterns or high-frequency textures.
In this paper, we propose a framework to learn spatially adaptive degradation representations of blurry images. A novel joint image reblurring and deblurring learning process is presented to improve the expressiveness of degradation representations. 
To make learned degradation representations effective in reblurring and deblurring, we propose a Multi-Scale Degradation Injection Network (MSDI-Net) to integrate them into the neural networks. With the integration, MSDI-Net can handle various and complicated blurry patterns adaptively. 
Experiments on the GoPro and RealBlur datasets demonstrate that our proposed deblurring framework with the learned degradation representations outperforms state-of-the-art methods with appealing improvements.

**Keywords**: Image Deblurring, Degradation Representations
