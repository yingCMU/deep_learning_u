- Anaconda comes with Conda, a package and environment manager built specifically for data science. Install the Python 3 version of Anaconda appropriate for your operating system.
```
conda create -n style-transfer python=3  # creates a new environment with Python 3. This environment will hold all the packages you need for the style transfer code.
conda activate style-transfer #enters the environment
conda install tensorflow scipy pillow #Next, we install TensorFlow, SciPy, Pillow (which is an image processing library), and moviepy.
pip install moviepy
python -c "import imageio; imageio.plugins.ffmpeg.download()" #The last line here installs ffmpeg, an application for converting images and videos.
```

Note: If you get an error on the last command that ends in
```
RuntimeError: imageio.ffmpeg.download() has been deprecated. Use 'pip install imageio-ffmpeg' instead.'
```
just run the command `pip install imageio-ffmpeg`  You may need to restart the terminal and reactivate the environment for the command to complete.
