# UNet Practice

Study of the original "U-Net: Convolutional Networks for Biomedical Image
Segmentation" paper by Brox et al in PyTorch.

## How to run this code

This codebase was developed to run on my Ubuntu-based workstation equipped with
32GiB RAM and an Nvidia GTX 1080 GPU. **This means that the training loop and much
of the code won't work if you try to run this code as-is on a non-CUDA-capable
device.**

### Requisite Dependencies

- Python 3.11.5
- PyTorch 2.X (see `pyproject.toml` for exact version)
- [Poetry](https://python-poetry.org/)
- A CUDA-capable device
- Git

### Build/Run Steps

```sh
# 1.  Clone this repo onto a CUDA-capable device 
$ git clone git@github.com:dwrodri/unet-practice.git
# 2. Go to project root
$ cd unet-practice
# 3.  Use Poetry to fetch dependencies and build a virtual env
$ poetry install
# 4. Run the script with training image folder and mask folder as args 
$ poetry run python3 main.py /path/to/images /path/to/masks
```

### What happens when you run this code?

This code fits a UNet on the VGG Pets Dataset so that it can perform the task of
creating a segmentation mask of the "pet" in the image. Each time that the training
loss improves, we write a sample of 10 generated masks to a `demo/` folder. Once
the training epochs finish (50 by default), we serialize the Model object to a file
called `unet_pets.pkl`. Finally we write 64 generated masks to the same `demo/` folder
so the model can be evaluated subjectively.

For convenience the images written to the demo folder are side-by-side comparison stacks,
with the sample input above the generated mask and the ground truth below.

### Code assumptions

- As mentioned earlier, **you need to run this on a CUDA-capable device to run this code as-is**.
- My PyTorch Dataset prep code expects two folders: one full of sample images and another full of segmentation masks for those images
- The sample image and mask image filename **must be the same** in order for the images to be paired properly.
- All sample images must use the `.png` or `.jpg` suffix, not `.jpeg`
- All ground truth masks must use `.png`

## Content Resources

- [Original Paper](https://arxiv.org/pdf/1505.04597.pdf)
- [Tutorial in PyTorch](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)
- [Tutorial in Keras, it changes up implementation to improve performance](https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/)
- [TGS Salt Identification Dataset](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data)
- [Oxford VGG Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
