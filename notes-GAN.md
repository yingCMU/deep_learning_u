# GAN

## Papers
- 2014 paper-[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [2016 Improved Techniques for Training GANs](https://video.udacity-data.com/topher/2018/November/5bea0c6a_improved-training-techniques/improved-training-techniques.pdf)


## Applications
- StackGAN realistic image synthesis: https://arxiv.org/abs/1612.03242
- iGAN interactive image generation: https://github.com/junyanz/iGAN
- CartoonGAN: https://video.udacity-data.com/topher/2018/November/5bea23cd_cartoongan/cartoongan.pdf
- You'll learn much more about Pix2Pix and CycleGAN formulations
- [Some cool applications of GAN](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900)
- [tulip generation]: The tulip generation model was created by the artist Anna Ridler, and you can read about her data collection method and inspiration in this [article](https://www.fastcompany.com/90237233/this-ai-dreams-in-tulips).

## How it works
Generator: takes random noise as input, run the noise through a differentiable function to reshape it to have recognizable structure. THe output is a realistic image.  The goal is for these images to be fair samples from the distribution over real data.
discriminator: learns to guide the generator. It is shown real data half of the time, and image from generator the other half of the time. 1 to real images, 0 to fake images, Generator tries to generate images that D assign one.

### Equilibria
Equilibria: non player can improve their strategy, given the other player don't change their strategy.
The saddle point: one player's max point & the other player's min point

But we may not able to find the equilibrium. We usually train GAN by running two optimization algorithms simultaneously.
A common failure for GAN is when data contain multiple clusters, then the generator will generate one cluster, and discriminator would learn to reject that cluster as being fake. THen the generator will learn generate another cluster.
We would prefer to have an algorithm that reliably finds the equilibrium where the generator samples from all the clusters simultaneously.
That is the key reserach problem in GAN, designing an algorithm for finding the equilibrium of a game involving high-demensional, continuous, non-convex cost functions

### Improved Training Techniques for GANs

- paper - [Improved Techniques for Training GANs](https://video.udacity-data.com/topher/2018/November/5bea0c6a_improved-training-techniques/improved-training-techniques.pdf)
- [lecture video](https://classroom.udacity.com/nanodegrees/nd101/parts/2ea78ff8-befd-4046-b06e-5327871b0748/modules/72f47c70-f4f0-49f8-a67a-a3562f0bd7ac/lessons/de9a07cd-bfb4-4d09-a305-2f20f158b965/concepts/8aa80681-3cee-421a-af6f-859e8205d11a)
