
![alt text](images/NHLBI_Standard_Sig_Logo_RGB.png)
___
# NHLBI, 2022 Fall

Welcome to the 2022 course!

This offering is for Division of Intramural (DIR), NHLBI in Fall, 2022. 

Based on the feedbacks from the last offering, this time we will split the course into two parts. The first part: **Deep Learning Fundamentals** consists of 8 lectures. We will start from basics of neural networks, introduce the loss function, optimization and how to setup and manage the training session. The next section is the convolutional neural network for imaging and vision tasks. We will then learn the recurrent neural network (RNN) for the sequence data. 

The second part: **Deep learning Advances** consists of 6 lectures. We will start at the attention mechanism and transformer models (BERT, GPT family etc.). In particular, the natural language processing applications using large deep learning models will be reviewed. We will then learn the generative model and in details the GAN (generative adversarial network). The technique to visualize the neural network is introduced to help understand how and why the neural network works. The course will end with a focus on how to handle "small dataset" usecase, as in many practical applications, we may not be able to acquire large labelled dataset. Three techniques are introduced, transfer learning, meta learning and contrastive learning (as the more recent development of self-supervised learning).

For the NHLBI DIR community, the teaching objectives are:

    * Introduce the basics of deep learning
    * Present in-math how DL model works
    * Provide practices to build your own model
    * Grow interest and improve community awareness
    * Prepare trainees and fellows for DL related jobs

For the student point of view, you will gradually learn the concepts and algorithms behind the deep neural network and master the tools required to build and train models. For every lecture, it comes with a reading list to broaden the understanding. 

We will use [Pytorch](www.pytorch.org). So it is a good idea to get yourself familiar with this [package](https://pytorch.org/tutorials/).

## Prerequisites

Please review the mathematics for deep learning and learn basic Python and Numpy. 

* [mathematics for deep learning](http://cs229.stanford.edu/summer2020/cs229-linalg.pdf)
* [Python, a more comprehensive book](https://cfm.ehu.es/ricardo/docs/python/Learning_Python.pdf)
* [Python Crash Course, one of the easiest tutorial](https://www.programmer-books.com/wp-content/uploads/2018/06/Python%20Crash%20Course%20-%20A%20Hands-On,%20Project-Based%20Introduction%20to%20Programming.pdf)
* [Numpy](https://cs231n.github.io/python-numpy-tutorial/)
* [Pytorch tutorial](https://pytorch.org/tutorials/)
* [Debug python program using VSCode](https://www.youtube.com/watch?v=W--_EOzdTHk)

## Instructors

* Hui Xue, hui.xue@nih.gov

## Schedule

Part 1:
Starting on the week of Sep 12, 2022

* Lecture, every Wed, 11:00am-12:30pm, US EST time
    - Link to be added

* Q&A session, every Friday, 11:00am-12:00pm, US EST time
    - Link to be added

Part 2:
Starting on the week of Jan 9, 2023

* Lecture, every Wed, 11:00am-12:30pm, US EST time
    - Link to be added

## Assignments

Three assignments will be provided for the part 1 of the course. These assignments combine the multi-choice questions and coding sampling. They are designed to help you understand the course content and practise skills to build models. 

Based on the feedback collected last year, the solutions of assignments will be provided to you two weeks after the release of questions ^(). The purpose is to encourage independent work, but still limit the time effort one may spend on the questions.

## Syllabus

## Part 1: Deep learning fundamentals

### **Prologue**

![L0 Intro](images/lectures/L0.png)

**Why do we want to spend hours in learning deep learning (DL)?** 
I can articulate one reason: Deep Learning is a set of key technique which can be applied to many fields, from mobile phone to medical imaging, from robotics to online shopping, from new drug discovery to genomics. What is really amazing to me is that in this wave of technological revolution, the **same** set of technique, deep neural network, is solving many challenging problems which are drastically different. Yes, it is the same set of algorithms, software toolboxes and knowledge base are applied, reporting state-of-the-art performance.

This makes learning deep learning rewardable, because you will master something which can be widely applied and mostly likely will stay that way in the decades to come. According to ARK's research, deep learning will add $30 trillion to the global equity market capitalization during the next 15-20 years*. No something which should be ignored!

However, there are difficulties along the way. Often, more than superficial level of understanding of DL is required, if you want to find a notch to apply DL in your field and if no one has done this before you. There will not be pre-trained models which you can download. One has to understand the problem and design the model, invent new loss functions and put all pieces together to build a DL solution. Your solution needs to prove its value in deployment and gets better over time.

This course is to equip you with required knowledge to understand and apply DL by teaching how the deep neural network models work and by reviewing many DL architectures and applications. My hope is after this learning process, domain experts will feel confident to apply DL to their specific problems and datasets.

#### Video

#### Slides

#### Suggested Reading

* \*For big pictures and where DL can fit, [Ark Big Idea](https://www.ark-bigideas.com/2022/en/pages/download)
* Artificial Intelligence Index Report 2022, [AI report](https://aiindex.stanford.edu/wp-content/uploads/2022/03/2022-AI-Index-Report_Master.pdf)
___
### **Lecture 1**

![L1 Intro](images/lectures/L1.png)

We start by motivating the deep learning for its broad applicability and future growth, and then introduce deep learning as a data driven approach. The basic terminology of neural network are reviewed. We set the stage for future discussion to introduce the binary and multi-class classification problems and the multi-layer perceptron (MLP) network. Other topics covered in this lecture include matrix broadcasting, universal approximation, logits, activation function etc.

#### Video

#### Slides

#### Suggested Reading

The same three authors wrote these two papers at the beginning of DL revolution and now. It is interesting to read and compare them.

* [Deep learning, Nature volume 521, 436–444 (2015)](https://www.nature.com/articles/nature14539.pdf)
* [Deep learning for all](https://dl.acm.org/doi/pdf/10.1145/3448250)
* [Multilayer perception, MLP](http://deeplearning.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)
___
### **Lecture 2**

![L2 Intro](images/lectures/L2.png)

This lecture introduces the concept of loss function to evaluate how well our model fits the data. The process to adjust model parameters to fit the data is called optimization. Gradient descent is a common algorithm used to optimize the model parameters, given the input dataset. This process is the training. We will review different training algorithms and introduce the concepts of training and testing. To measure our model performance in training and testing datasets, the bias and variance of model should be estimated. Other concepts introduced in this lecture include regularization, under-fitting, over-fitting, batch and mini-batch etc.

#### Video

#### Slides

#### Suggested Reading

* [Notes on optimization](https://cs231n.github.io/optimization-1/)
* [Loss functions in deep learning by Artem Oppermann](./Loss Functions in Deep Learning _ MLearning.ai.pdf)
* [Bias and variance](https://www.bradyneal.com/bias-variance-tradeoff-textbooks-update)
___
### **Lecture 3**

![L3 Intro](images/lectures/L3.png)

The key step to train a model is to follow the negative gradient direction to reduce the loss. But how do we compute the gradient direction? Through a process called the back propagation or backprop in short. This lecture discusses the backprop in detail. Backprop is based on two ideas: chain rule of derivative and divide-and-conquer. It allows us to compute complex derivative from loss to every learnable parameters in the model. 

We will not review GPU devices for deep learning in lectures. Please review two documents in this week's reading list.

#### Video

#### Slides

#### Suggested Reading

* [How backprop works](http://neuralnetworksanddeeplearning.com/chap2.html)
* [Derivatives of tensor](http://cs231n.stanford.edu/handouts/derivatives.pdf)
* [Autograd in Pytorch](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
* [GPU in Pytorch](https://www.run.ai/guides/gpu-deep-learning/pytorch-gpu/)
* [GPU for deep learning](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)
___
### **Lecture 4**
An interactive session to reveal the math behind the neural network

This lecture will review the math behind backprop and gradient update. In particular, we will derive the gradients for a binary and multi-class classification multi-layer perception and explain in details why the algorithm is organized in the particular way. The interactive python coding will be conducted to demo how to build up numerical models and how to debug the implementation. 

#### Video

#### Slides

#### Suggested Reading and demos

[Download the linear regression demo](slides/linear_regression.py)

Drs. Eric Leifer and James Troendle from the NHLBI drafted this excellent notes about the derivation of loss gradient. Please review.
[Download the notes for gradient derivation](2021-11-15_gradient_matrices_Assignment_1.pdf)

### **Assignment 1**

[Download the Assignment 1](https://gadgetrondata.blob.core.windows.net/dlcc/dlcc_assignment1.zip)

In this assignment, you will be asked to implement the multi-layer perceptron model and cross-entropy loss. The coding problem will require the implementation for both forward pass and backprop. The gradient descent is used to train the model for higher classification accuracy. We will not use deep learning framework in this assignment, but will use Python+Numpy combination. The goal is to make sure the thorough understanding of mathematics and numeric technique necessary for a classic neural network. Also, it is to encourage one to get familiar with python coding.

This assignment introduces the regression test for model training and evaluation. The [Pytest](https://docs.pytest.org/en/6.2.x/) is used for the regression test purpose.

___
### **Lecture 5**

![L4 Intro](images/lectures/L4.png)

This lecture will finish our discussion on different optimization algorithms. We will introduce a few new methods and compare their pros and cons. The concept of hyper-parameter is explained, where a very important one is the learning rate. Different learning rate scheduling strategies are discussed in this lecture and help boost training performance. To search a good configuration of hyper-parameters, we will discuss coarse-to-fine, hyper-band and Bayesian methods. We close the lecture by discussing bag of tricks to set up the training process and cross-validation.

#### Video

#### Slides

#### Suggested Reading

* [Optimization in deep learning](https://ruder.io/optimizing-gradient-descent/)
* [Learning rate scheduler in Pytorch](https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling)
* [One-cycle learning rate scheduler](https://arxiv.org/pdf/1803.09820.pdf)
* [One-cycle learning rate scheduler, post](https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6)
* [Hyper-parameter searching](https://arxiv.org/pdf/2003.05689.pdf)
* [Set up training, chapter 40, 41, 42](https://d2wvfoqc9gyqzf.cloudfront.net/content/uploads/2018/09/Ng-MLY01-13.pdf)
* [DL experiment management](./A quick guide to managing machine learning experiments _ by Shashank Prasanna _ Towards Data Science.pdf)

___
### **Lecture 6**

![L5 Intro](images/lectures/L5.PNG)

This lecture continues our discussion on training setup, with focus on handling data mismatching between training and test sets. The meaning and strategy to conduct error analysis are introduced. After finishing the training section, we discuss the method for data pre-processing and how to initialize the model parameters. The final section of this lecture introduces the deep learning debugging and iteration for model training. Tools for debugging are demonstrated.

#### Video

#### Slides

#### Suggested Reading

* [Data mismatching](https://insights.sei.cmu.edu/blog/detecting-mismatches-machine-learning-systems/)
* [Data pre-processing](https://serokell.io/blog/data-preprocessing)
* [Data transformation in TorchVision](https://pytorch.org/vision/stable/transforms.html)
* [Checklist to debug NN](https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21)

___
### **Assignment 2**

[Download the Assignment 2](https://gadgetrondata.blob.core.windows.net/dlcc/assignment2.zip)

In this assignment, you will study the autodiff in Pytorch, which is essential to understand how DL models work. Then you will implement a MLP model using Pytorch on the Cifar10 dataset. Another important aspects in this assignment is the experimental management, using the [wandb](https://wandb.ai/site).

___
### **Lecture 7**

![L6 Intro](images/lectures/L6.png)

This lecture starts the convolutional neural network (CNN) by introducing the convolution operation and its application on image. Different variants of convolution is discussed, including stride, transpose, dilated CONV, 1D and 3D CONV, padding, and pooling. Image interpolation layer is introduced with other methods to up/downsample images. The batch normalization is discussed with other feature normalization methods. Two CNN architectures are analyzed, LeNet-5 and AlexNet, in the history of ImageNet challenge.

#### Video

#### Slides

#### Suggested Reading

* [CONV and its variants](https://arxiv.org/pdf/1603.07285.pdf)
* [CNN explanation](https://poloclub.github.io/cnn-explainer/)
* [Introduction for batch norm, layer norm, group norm etc.](./Normalization Techniques in Deep Neural Networks _ by Aakash Bindal _ Techspace _ Medium.pdf)
* [Overview of NN feature normalization](https://arxiv.org/pdf/2009.12836.pdf)
* [ImageNet Winning CNN Architectures](https://www.kaggle.com/getting-started/149448)

___
### **Lecture 8**

![L7 Intro](images/lectures/L7.png)

With the basics of CONV and CNN introduced in last lecture, we continue to go through the history of ImageNet competition and reviewed winning architectures until 2017 and go beyond for very latest developments, including ResNet and its variants, group convolution, mobile net, efficient net. We can learn key ideas to design and refine the CNN architectures. The second part of this lecture discusses applications of CNN, including two-stage and one-stage object detection, landmark detection, U-net for segmentation, denoising CNN and super-resolution CNN. 

Network compression is not discussed in the lecture. But you are encouraged to read more on this topic.

#### Video

#### Slides

#### Suggested Reading

* [ResNet paper](https://arxiv.org/abs/1512.03385)
* [ResNet with batch norm](https://arxiv.org/abs/1603.05027)
* [Introduction for batch norm, layer norm, group norm etc.](./Normalization Techniques in Deep Neural Networks _ by Aakash Bindal _ Techspace _ Medium.pdf)
* [Mobile, shuffle, effnet](https://towardsdatascience.com/3-small-but-powerful-convolutional-networks-27ef86faa42d)
* [One-stage object detector](https://www.jeremyjordan.me/object-detection-one-stage/)
* [ResUnet](https://arxiv.org/pdf/1711.10684.pdf)
* [Intro for network compression](https://towardsdatascience.com/how-to-compress-a-neural-network-427e8dddcc34)

___
### **Assignment 3**

[Download the Assignment 3](https://gadgetrondata.blob.core.windows.net/dlcc/assignments3.zip)

In this assignment, you will practice to build CNN models. Two new datasets are introduced, cifar10 and carvana datasets. You will build classficiation CNN models on cifar10 datasets and build a segmentatin U-net model on carvana datasets. This is an important assignment. Please give a try!