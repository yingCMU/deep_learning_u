# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
# If 𝑀>2  (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result.
def CrossEntropy(yHat, y):
    if y == 1:
      return -log(yHat)
    else:
      return -log(1 - yHat)
# pytorch: https://pytorch.org/docs/stable/nn.html#crossentropyloss
