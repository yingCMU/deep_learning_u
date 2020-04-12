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
https://keras.io/examples/mnist_mlp/
https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

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
