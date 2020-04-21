# CNN
## resources
- [Stanford lecture notes](https://cs231n.github.io/neural-networks-1/)
## normalization
https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor
Normalizing image inputs
Data normalization is an important pre-processing step. It ensures that each input (each pixel value, in this case) comes from a standard distribution. That is, the range of pixel values in one input image are the same as the range in another image. This standardization makes our model train and reach a minimum error, faster!

Data normalization is typically done by subtracting the mean (the average of all pixel values) from each pixel, and then dividing the result by the standard deviation of all the pixel values. Sometimes you'll see an approximation here, where we use a mean and standard deviation of 0.5 to center the pixel values. Read more about the Normalize transformation in PyTorch.

The distribution of such data should resemble a [Gaussian](https://mathworld.wolfram.com/GaussianFunction.html) function centered at zero. For image inputs we need the pixel numbers to be positive, so we often choose to scale the data in a normalized range [0,1].

## MLP MNIST
- See [note_mnist.md](./note_mnist.md)
- CNN can perform much better that MLP in image classification. MNIST is an exception in that all images are roughly the same size and are centered in 28*28 pixel grid (heavily pre-processed). If instead the digit appear any where within the grid, some digit quite small other quite large, it would be more challenging task for MLP. In real world messy images, CNN beat MLP. CNN understands spacial proximity : http://yann.lecun.com/exdb/mnist/

## Image Classification Steps
![alt text](./images/![alt text](./images/validation_set.png "validation") "Steps")

## Filters
To detect changes in intensity in an image, you’ll be using and creating specific image filters that look at groups of pixels and react to alternating patterns of dark/light pixels. These filters produce an output that shows edges of objects and differing textures.
- High and low frequency. Frequency in images is a rate of change. But, what does it means for an image to change? Well, images change in space, and a high frequency image is one where the intensity changes a lot. And the level of brightness changes quickly from one pixel to the next. A low frequency image may be one that is relatively uniform in brightness or changes very slowly.
  - High-frequency components also correspond to the edges of objects in images, which can help us classify those objects.
- kernels: Edge Handling,
Kernel convolution relies on centering a pixel and looking at it's surrounding neighbors. So, what do you do if there are no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are listed below. It’s most common to use padding, cropping, or extension. In extension, the border pixels of an image are copied and extended far enough to result in a filtered image of the same size as the original image.

  - Extend The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.

  - Padding The image is padded with a border of 0's, black pixels.

  - Crop Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being slightly smaller, with the edges having been cropped.


## Convolutional Layer
The convolutional layer is produced by applying a series of many different image filters, also known as convolutional kernels, to an input image.
In the example shown, 4 different filters produce 4 differently filtered output images. When we stack these images, we form a complete convolutional layer with a depth of 4!
![alt text](./images/![alt text](./images/conv_layer.png "validation") "Steps")

## visualizing conv layer :
- https://setosa.io/ev/image-kernels/
-[conv_visualization](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/conv_visualization.ipynb)
### Learning
In the code you've been working with, you've been setting the values of filter weights explicitly, but neural networks will actually learn the best filter weights as they train on a set of image data.
In practice, you'll also find that many neural networks learn to detect the edges of images because the edges of object contain valuable information about the shape of an object.

- dense layers are fully connected, meaning each node is connected to every node in the previous layer
- convolutional layers are locally connected, meaning their nodes are connected to only a small subset of the previous layers' nodes. CNN determins what kind of pattern it needs to detect based on the loss function.
### Pooling
With stride & window size
- max pooling:
- [average pooling](https://pytorch.org/docs/stable/nn.html#avgpool2d):average pixel values in a given window size. So in a 2x2 window, this operation will see 4 pixel values, and return a single, average of those four values, as output!
  - This kind of pooling is typically not used for image classification problems because maxpooling is better at noticing the most important details about edges and other features in an image, but you may see this used in applications for which smoothing an image is preferable.

code:  conv_visualization.ipynb & maxpooling_visualization.ipynb


### Alternatives to Pooling
It's important to note that pooling operations do throw away some image information. That is, they discard pixel information in order to get a smaller, feature-level representation of an image. This works quite well in tasks like image classification, but it can cause some issues.

Consider the case of facial recognition. When you think of how you identify a face, you might think about noticing features; two eyes, a nose, and a mouth, for example. And those pieces, together, form a complete face! A typical CNN that is trained to do facial recognition, should also learn to identify these features. Only, by distilling an image into a feature-level representation, you might get a weird result:

Given an image of a face that has been photoshopped to include three eyes or a nose placed above the eyes, a feature-level representation will identify these features and still recognize a face! Even though that face is fake/contains too many features in an atypical orientation.
So, there has been research into classification methods that do not discard spatial information (as in the pooling layers), and instead learn to spatial relationships between parts (like between eyes, nose, and mouth).

### capsule network
One such method, for learning spatial relationships between parts, is the capsule network.

- Capsule Networks provide a way to detect parts of objects in an image and represent spatial relationships between those parts. This means that capsule networks are able to recognize the same object, like a face, in a variety of different poses and with the typical number of features (eyes, nose , mouth) even if they have not seen that pose in training data.

Capsule networks are made of parent and child nodes that build up a complete picture of an object.

### What are Capsules?
Capsules are essentially a collection of nodes, each of which contains information about a specific part; part properties like width, orientation, color, and so on. The important thing to note is that each capsule outputs a vector with some magnitude and orientation.

Magnitude (m) = the probability that a part exists; a value between 0 and 1.
Orientation (theta) = the state of the part properties.
These output vectors allow us to do some powerful routing math to build up a parse tree that recognizes whole objects as comprised of several, smaller parts!

The magnitude is a special part property that should stay very high even when an object is in a different orientation, as shown below.
- https://cezannec.github.io/Capsule_Networks/
- implementation of a capsule network in PyTorch, at this [github repo](https://github.com/cezannec/capsule_net_pytorch)
- Paper [Dynamic Routing Between Capsules](https://video.udacity-data.com/topher/2018/November/5bfdca4f_dynamic-routing/dynamic-routing.pdf)

## Architecture
![alt text](./images/![alt text](./images/cnn_basic_arch.png "validation") "Arch")
- increasing depth: conv layers will make the array deeper as it passes through the network.
- max pooling layer will be used to decrease the XY dimensions
- as the network gets deeper, it's actually extracting more and more complex patterns features.And it's actually dicarding some spatial information about features,like a smooth background, that does not help in indentifying an image.
![alt text](./images/![alt text](./images/increasing_depth.png "validation") "Arch")

- How might you define a Maxpooling layer, such that it down-samples an input by a factor of 4?  The best choice would be to use a kernel and stride of 4, so that the maxpooling function sees every input pixel once, but any layer with a stride of 4 will down-sample an input by that factor.
- both stride and padding can decide the XY dimension of output
- The filter itself is 3-d dimensions, the 3 channel's sum is added together:
![alt text](./images/![alt text](./images/3d_filter.png "validation") "Arch")

## Feature Vecotr
feature level representation:
![alt text](./images/![alt text](./images/feature_vector.png "validation") "Arch")



### pytorch
- Convolutional Layers
We typically define a convolutional layer in PyTorch using nn.Conv2d, with the following parameters, specified:

`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`
in_channels refers to the depth of an input. For a grayscale image, this depth = 1
out_channels refers to the desired depth of the output, or the number of filtered images you want to get as output
kernel_size is the size of your convolutional kernel (most commonly 3 for a 3x3 kernel)
stride and padding have default values, but should be set depending on how large you want your output to be in the spatial dimensions x, y
Read more about Conv2d in the documentation.

- Pooling Layers
Maxpooling layers commonly come after convolutional layers to shrink the x-y dimensions of an input, read more about pooling layers in PyTorch, here.

###Padding
is just adding a border of pixels around an image. In PyTorch, you specify the size of this border.

Why do we need padding?

When we create a convolutional layer, we move a square filter around an image, using a center-pixel as an anchor. So, this kernel cannot perfectly overlay the edges/corners of images. The nice feature of padding is that it will allow us to control the spatial size of the output volumes (most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same).

The most common methods of padding are padding an image with all 0-pixels (zero padding) or padding them with the nearest pixel value. You can read more about calculating the amount of padding, given a kernel_size, [here](https://cs231n.github.io/convolutional-networks/#conv).


## Formula: Number of Parameters in a Convolutional Layer
The number of parameters in a convolutional layer depends on the supplied values of filters/out_channels, kernel_size, and input_shape. Let's define a few variables:

K - the number of filters in the convolutional layer
F - the height and width of the convolutional filters
D_in - the depth of the previous layer

Notice that K = out_channels, and F = kernel_size. Likewise, D_in is the last value in the input_shape tuple, typically 1 or 3 (RGB and grayscale, respectively).

Since there are F*F*D_in weights per filter, and the convolutional layer is composed of K filters, the total number of weights in the convolutional layer is K*F*F*D_in. Since there is one bias term per filter, the convolutional layer has K biases. Thus, the number of parameters in the convolutional layer is given by K*F*F*D_in + K.

## Formula: Shape of a Convolutional Layer
The shape of a convolutional layer depends on the supplied values of kernel_size, input_shape, padding, and stride. Let's define a few variables:

K - the number of filters in the convolutional layer
F - the height and width of the convolutional filters
S - the stride of the convolution
P - the padding
W_in - the width/height (square) of the previous layer
Notice that K = out_channels, F = kernel_size, and S = stride. Likewise, W_in is the first and second value of the input_shape tuple.

The depth of the convolutional layer will always equal the number of filters K.

The spatial dimensions of a convolutional layer can be calculated as: (W_in−F+2P)/S+1


## Flattening
Part of completing a CNN architecture, is to flatten the eventual output of a series of convolutional and pooling layers, so that all parameters can be seen (as a vector) by a linear classification layer. At this step, it is imperative that you know exactly how many parameters are output by a layer.

For the following quiz questions, consider an input image that is 130x130 (x, y) and 3 in depth (RGB). Say, this image goes through the following layers in order:
```
nn.Conv2d(3, 10, 3)
nn.MaxPool2d(4, 4)
nn.Conv2d(10, 20, 5, padding=2)
nn.MaxPool2d(2, 2)
```
After going through all four of these layers in sequence, what is the depth of the final output?

## Preprocessing Images
1. resizing: standardize image size
2. normalization
3. convert to tensor data type


## Data augmentation
- rotation invariant
- scale invariant
rotation, translation of an image object to augment training data and help CNN to learn better, avoid over-fitting as well
- pytorch has transformation package: torchvision.transforms
- notebook code [here](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/cifar-cnn/cifar10_cnn_augmentation.ipynb)

## Breakthroughs in CNN Archtiecutre
-  AlexNet [paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf): 11x11
- Read more about [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) here: 3x3
- The ResNet paper can be found [here](https://arxiv.org/pdf/1512.03385v1.pdf). ResNet add connect to skip layers in their deep arch, so the gradient has a shorter route to travel.

Resources:
- Here's the [Keras](https://keras.io/applications/) documentation for accessing some famous CNN architectures.
- Read this [detailed treatment](http://neuralnetworksanddeeplearning.com/chap5.html) of the vanishing gradients problem. The deeper the layers, the more likely the signal get weakended before it gets where it needs to go,
- Here's a GitHub [repository](https://github.com/jcjohnson/cnn-benchmarks) containing benchmarks for different CNN architectures.
- Visit the ImageNet Large Scale Visual Recognition Competition (ILSVRC) [website](http://www.image-net.org/challenges/LSVRC/).

## Visualizing CNNs

- Here's a section from the [Stanford's CS231n](https://cs231n.github.io/understanding-cnn/) course on visualizing what CNNs learn.
- Check out [this demonstration](https://experiments.withgoogle.com/what-neural-nets-see) of a cool [OpenFrameworks app](https://openframeworks.cc/) that visualizes CNNs in real-time, from user-supplied video!
- Here's a [demonstration](https://www.youtube.com/watch?v=AgkfIQ4IGaM&t=78s) of another visualization tool for CNNs. If you'd like to learn more about how these visualizations are made, check out this [video](https://www.youtube.com/watch?v=ghEmQSxT6tw&t=5s).
- Read this Keras [blog post](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html) on visualizing how CNNs see the world. In this post, you can find an accessible introduction to Deep Dreams, along with code for writing your own deep dreams in Keras. When you've read that:

  - Also check out this [music video](https://www.youtube.com/watch?v=XatXy6ZhKZw) that makes use of Deep Dreams (look at 3:15-3:40)!
  - Create your own Deep Dreams (without writing any code!) using this [website](https://deepdreamgenerator.com/).

- If you'd like to read more about interpretability of CNNs:

  - Here's an [article](https://openai.com/blog/adversarial-example-research/) that details some dangers from using deep learning models (that are not yet interpretable) in real-world applications.
  - There's a lot of active research in this area. [These authors](https://arxiv.org/abs/1611.03530) recently made a step in the right direction - Understanding deep learning requires rethinking generalization

### Visualizing CNN example
- Paper [Visualizing and Understanding Convolutional Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)
- [Video](https://www.youtube.com/watch?v=ghEmQSxT6tw)


## Applications
- [WaveNet model](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio).

Why train an A.I. to talk, when you can train it to sing ;)? In April 2017, researchers used a variant of the WaveNet model to generate songs. The original [paper-a-neural-parametric-singing-synthesizer](https://arxiv.org/abs/1704.03809) and demo can be found here.
- CNNs for text classification.
[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
You might like to sign up for the author's Deep Learning [Newsletter](https://www.getrevue.co/profile/wildml)!

- Facebook's novel CNN approach for language translation that achieves state-of-the-art accuracy at nine times the speed of RNN models.https://engineering.fb.com/ml-applications/a-novel-approach-to-neural-machine-translation/

- Play [Atari games- Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning) with a CNN and reinforcement learning. You can download the [code](https://sites.google.com/a/deepmind.com/dqn/) that comes with this paper.

If you would like to play around with some beginner code (for deep reinforcement learning), you're encouraged to check out Andrej Karpathy's [post](http://karpathy.github.io/2016/05/31/rl/).

- Play pictionary with a CNN!

Also check out all of the other cool implementations on the A.I. Experiments [website](https://experiments.withgoogle.com/collection/ai). Be sure not to miss [AutoDraw](https://www.autodraw.com)!

- Read more about [AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far).

- Check out this [article](https://www.technologyreview.com/2017/04/28/106009/finding-solace-in-defeat-by-artificial-intelligence/), which asks the question: If mastering Go “requires human intuition,” what is it like to have a piece of one’s humanity challenged?

- Check out these really cool videos with drones that are powered by CNNs.

  - Here's an interview with a startup - Intelligent Flying Machines (IFM)https://www.youtube.com/watch?v=AMDiR61f86Y.
  - Outdoor autonomous navigation is typically accomplished through the use of the global positioning system ([GPS](https://www.droneomega.com/gps-drone-navigation-works/)), but here's a [demo](https://www.youtube.com/watch?v=wSFYOw4VIYY) with a CNN-powered autonomous drone.

- If you're excited about using CNNs in self-driving cars, you're encouraged to check out:

this series of [blog posts](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/) that details how to train a CNN in Python to produce a self-driving A.I. to play Grand Theft Auto V.
- more

  - Some of the world's most famous paintings have been t[urned into 3D](https://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1) for the visually impaired. Although the article does not mention how this was done, we note that it is possible to use a CNN to [predict depth](https://cs.nyu.edu/~deigen/depth/) from a single image.
  - Check out this [research](https://ai.googleblog.com/2017/03/assisting-pathologists-in-detecting.html) that uses CNNs to localize breast cancer.
  - CNNs are used to save [endangered species](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)!
  - An app called [FaceApp](https://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/) uses a CNN to make you smile in a picture or change genders.
