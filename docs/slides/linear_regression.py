
##################################################
## Deep learning crash course, demo for setup
##################################################
## Description : A demo file to do simple linear regression
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

# this file is adapted from https://colab.research.google.com/drive/1HS3qbHArkqFlImT2KnF5pcMCz7ueHNvY?usp=sharing#scrollTo=qE__Ygl2c4IS

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# disable the interactive plotting
matplotlib.use("agg")

# ------------------------
# declare the model

class Linear:
    def __init__(self, input_dim: int, num_hidden: int = 1):
        self.weights = np.random.randn(input_dim, num_hidden) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(num_hidden)
  
    def __call__(self, x):
        self.x = x
        output = x @ self.weights + self.bias
        return output

    def backward(self, gradient):
        self.weights_gradient = self.x.T @ gradient
        self.bias_gradient = gradient.sum(axis=0)
        self.x_gradient = gradient @ self.weights.T
        return self.x_gradient

    def update(self, lr):
        self.weights = self.weights - lr * self.weights_gradient
        self.bias = self.bias - lr * self.bias_gradient

# ------------------------
# declare the loss

class MSE:
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return ((y_true - y_pred) ** 2).mean()
  
    def backward(self):
        n = self.y_true.shape[0]
        self.gradient = 2. * (self.y_pred - self.y_true) / n
        return self.gradient

# ------------------------
# perform the optimization

def training(x, y_true, d, num_epochs=40, lr=0.1):

    loss = MSE()
    linear = Linear(d)
   
    for epoch in range(num_epochs):
        y_pred = linear(x)
        loss_value = loss(y_pred, y_true)

        if epoch % 2 == 0:
            print(f'Epoch {epoch}, loss {loss_value}')
            plt.plot(x, y_pred.squeeze(), label=f'Epoch {epoch}')

        gradient_from_loss = loss.backward()
        linear.backward(gradient_from_loss)
        linear.update(lr)

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    
    return linear
    
# ------------------------
# main function to train this model
def main():
    
    # ------------------------
    # set up the x, y
    n = 50
    d = 1
    x = np.random.uniform(-1, 1, (n, d))

    # y = 5x + 10
    weights_true = np.array([[5],])
    bias_true = np.array([10])

    y_true = x @ weights_true + bias_true
    print(f'x: {x.shape}, weights: {weights_true.shape}, bias: {bias_true.shape}, y: {y_true.shape}')    

    # plot the model prediction before fitting
    linear = Linear(d)
    y_initail_pred = linear(x)
    
    best_model = training(x, y_true, d, num_epochs=40, lr=0.1)

    y_best_pred = best_model(x)
    
    fig=plt.figure(figsize=[8, 8])
    plt.plot(x, y_true, marker='x', label='underlying function')
    plt.scatter(x, y_initail_pred, color='r', marker='.', label='before fitting')
    plt.scatter(x, y_best_pred, color='k', marker='.', label='after fitting')
    plt.legend()

    fig.savefig('linear_regression.jpg', dpi=300)

if __name__ == '__main__':
    main()