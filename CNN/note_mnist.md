# MLP MNIST
https://keras.io/examples/mnist_mlp/
https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
## Cross-Entropy Loss
In the PyTorch [documentation](https://pytorch.org/docs/stable/nn.html#crossentropyloss), you can see that the cross entropy loss function actually involves two steps:

- It first applies a softmax function to any output is sees
- Then applies [NLLLoss](https://pytorch.org/docs/stable/nn.html#nllloss); negative log likelihood loss
Then it returns the average loss over a batch of data. Since it applies a softmax function, we do not have to specify that in the forward function of our model definition, but we could do this another way.

### Another approach
We could separate the softmax and NLLLoss steps.

In the forward function of our model, we would explicitly apply a softmax activation function to the output, x.
```
# a softmax layer to convert 10 outputs into a distribution of class probabilities
x = F.log_softmax(x, dim=1)

return x
```
Then, when defining our loss criterion, we would apply NLLLoss
```
# cross entropy loss combines softmax and nn.NLLLoss() in one single class
# here, we've separated them
criterion = nn.NLLLoss()
```
This separates the usual criterion = nn.CrossEntropy() into two steps: softmax and NLLLoss, and is a useful approach should you want the output of a model to be class probabilities rather than class scores.
