# pytorch

## setup
- The easiest way to install PyTorch and torchvision locally is by following the instructions on the PyTorch site
https://pytorch.org/get-started/locally/
- You'll also need to install numpy and jupyter notebooks, the newest versions of these should work fine. Using the conda package manager is generally best for this,
`conda install numpy jupyter notebook`
[conda doc](https://conda.io/en/latest/)
- GPU: The final part of the series has a soft requirement of a GPU used to accelerate network computations. Even if you don't have a GPU available, you'll still be able to run the code and finish the exercises. PyTorch uses a library called CUDA to accelerate operations using the GPU. If you have a GPU that CUDA supports, you'll be able to install all the necessary libraries by installing PyTorch with conda. If you can't use a local GPU, you can use cloud platforms such as [AWS](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html), [GCP](https://cloud.google.com/gpu/), and [FloydHub](https://www.floydhub.com/) to train your networks on a GPU.
