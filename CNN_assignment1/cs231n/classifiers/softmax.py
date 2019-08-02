import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_examples = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_examples):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    scores = np.exp(scores)
    scores /= np.sum(scores)
    loss -= np.log(scores[y[i]])
    for j in range(num_classes):
      if j == y[i]:
        dW[:,y[i]] += X[i] * (scores[j] -1)
      else:
        dW[:,j] += X[i] * scores[j]
  
  loss /= num_examples
  loss += reg * np.sum(W * W)
  dW /= num_examples
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_examples = X.shape[0]
  scores = X.dot(W)
  scores = (scores.T - np.max(scores,1)).T
  scores = np.exp(scores)
  scores = (scores.T / np.sum(scores,1)).T
  loss = np.sum(-np.log(scores[np.arange(num_examples),y]))
  scores[np.arange(num_examples),y] -= 1
  dW = (X.T).dot(scores)
  loss /= num_examples
  loss += reg * np.sum(W * W)
  dW /= num_examples
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

