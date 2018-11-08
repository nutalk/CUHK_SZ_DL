# Generative Adversarial Networks


Throughout most of this book, we've talked about how to make predictions.
In some form or another, we used deep neural networks learned mappings from data points to labels.
This kind of learning is called discriminative learning,
as in, we'd like to be able to discriminate between photos cats and photos of dogs. 
Classifiers and regressors are both examples of discriminative learning. 
And neural networks trained by backpropagation 
have upended everything we thought we knew about discriminative learning 
on large complicated datasets. 
Classification accuracies on high-res images has gone from useless 
to human-level (with some caveats) in just 5-6 years. 
We'll spare you another spiel about all the other discriminative tasks 
where deep neural networks do astoundingly well.

But there's more to machine learning than just solving discriminative tasks.
For example, given a large dataset, without any labels,
we might want to learn a model that concisely captures the characteristics of this data.
Given such a model, we could sample synthetic data points that resemble the distribution of the training data.
For example, given a large corpus of photographs of faces,
we might want to be able to generate a *new* photorealistic image 
that looks like it might plausibly have come from the same dataset. 
This kind of learning is called *generative modeling*. 

Until recently, we had no method that could synthesize novel photorealistic images. 
But the success of deep neural networks for discriminative learning opened up new possiblities.
One big trend over the last three years has been the application of discriminative deep nets
to overcome challenges in problems that we don't generally think of as supervised learning problems.
The recurrent neural network language models are one example of using a discriminative network (trained to predict the next character)
that once trained can act as a generative model. 


In 2014, a young researcher named Ian Goodfellow introduced [Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661) a clever new way to leverage the power of discriminative models to get good generative models. 
GANs made quite a splash so it's quite likely you've seen the images before. 
For instance, using a GAN you can create fake images of bedrooms, as done by [Radford et al. in 2015](https://arxiv.org/pdf/1511.06434.pdf) and depicted below. 

![](../img/fake_bedrooms.png)

At their heart, GANs rely on the idea that a data generator is good
if we cannot tell fake data apart from real data. 
In statistics, this is called a two-sample test - a test to answer the question whether datasets $X = \{x_1, \ldots x_n\}$ and $X' = \{x_1', \ldots x_n'\}$ were drawn from the same distribution. 
The main difference between most statistics papers and GANs  is that the latter use this idea in a constructive way.
In other words, rather than just training a model to say 'hey, these two datasets don't look like they came from the same distribution', they use the two-sample test to provide training signal to a generative model.
This allows us to improve the data generator until it generates something that resembles the real data. 
At the very least, it needs to fool the classifier. And if our classifier is a state of the art deep neural network.

As you can see, there are two pieces to GANs - first off, we need a device (say, a deep network but it really could be anything, such as a game rendering engine) that might potentially be able to generate data that looks just like the real thing. 
If we are dealing with images, this needs to generate images. 
If we're dealing with speech, it needs to generate audio sequences, and so on. 
We call this the *generator network*. The second component is the *discriminator network*. 
It attempts to distinguish fake and real data from each other. 
Both networks are in competition with each other. 
The generator network attempts to fool the discriminator network. At that point, the discriminator network adapts to the new fake data. This information, in turn is used to improve the generator network, and so on. 

**Generator**
* Draw some parameter $z$ from a source of randomness, e.g. a normal distribution $z \sim \mathcal{N}(0,1)$.
* Apply a function $f$ such that we get $x' = G(u,w)$
* Compute the gradient with respect to $w$ to minimize $\log p(y = \mathrm{fake}|x')$ 

**Discriminator**
* Improve the accuracy of a binary classifier $f$, i.e. maximize $\log p(y=\mathrm{fake}|x')$ and $\log p(y=\mathrm{true}|x)$ for fake and real data respectively.


![](../img/simple-gan.png)

In short, there are two optimization problems running simultaneously, and the optimization terminates if a stalemate has been reached. There are lots of further tricks and details on how to modify this basic setting. For instance, we could try solving this problem in the presence of side information. This leads to cGAN, i.e. conditional Generative Adversarial Networks. We can change the way how we detect whether real and fake data look the same. This leads to wGAN (Wasserstein GAN), kernel-inspired GANs and lots of other settings, or we could change how closely we look at the objects. E.g. fake images might look real at the texture level but not so at the larger level, or vice versa. 

Many of the applications are in the context of images. Since this takes too much time to solve in a Jupyter notebook on a laptop, we're going to content ourselves with fitting a much simpler distribution. We will illustrate what happens if we use GANs to build the world's most inefficient estimator of parameters for a Gaussian. Let's get started.


```python
from __future__ import print_function
import matplotlib as mpl
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import numpy as np

ctx = mx.cpu()
```

## Generate some 'real' data

Since this is going to be the world's lamest example, we simply generate data drawn from a Gaussian. And let's also set a context where we'll do most of the computation.

```python
X = nd.random_normal(shape=(1000, 2))
A = nd.array([[1, 2], [-0.1, 0.5]])
b = nd.array([1, 2])
X = nd.dot(X, A) + b
Y = nd.ones(shape=(1000, 1))

# and stick them into an iterator
batch_size = 4
train_data = mx.io.NDArrayIter(X, Y, batch_size, shuffle=True)
```

Let's see what we got. This should be a Gaussian shifted in some rather arbitrary way with mean $b$ and covariance matrix $A^\top A$.

```python
plt.scatter(X[:,0].asnumpy(), X[:,1].asnumpy())
plt.show()
print("The covariance matrix is")
print(nd.dot(A.T, A))
```

## Defining the networks

Next we need to define how to fake data. Our generator network will be the simplest network possible - a single layer linear model. This is since we'll be driving that linear network with a Gaussian data generator. Hence, it literally only needs to learn the parameters to fake things perfectly. For the discriminator we will be a bit more discriminating: we will use an MLP with 3 layers to make things a bit more interesting. 

The cool thing here is that we have *two* different networks, each of them with their own gradients, optimizers, losses, etc. that we can optimize as we please. 

```python
# build the generator
netG = nn.Sequential()
with netG.name_scope():
    netG.add(nn.Dense(2))

# build the discriminator (with 5 and 3 hidden units respectively)
netD = nn.Sequential()
with netD.name_scope():
    netD.add(nn.Dense(5, activation='tanh'))
    netD.add(nn.Dense(3, activation='tanh'))
    netD.add(nn.Dense(2))

# loss
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# initialize the generator and the discriminator
netG.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.initialize(mx.init.Normal(0.02), ctx=ctx)

# trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})
```

## Setting up the training loop

We are going to iterate over the data a few times. To make life simpler we need a few variables

```python
real_label = mx.nd.ones((batch_size,), ctx=ctx)
fake_label = mx.nd.zeros((batch_size,), ctx=ctx)
metric = mx.metric.Accuracy()

# set up logging
from datetime import datetime
import os
import time
```

## Training loop



```python
stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
for epoch in range(10):
    tic = time.time()
    train_data.reset()
    for i, batch in enumerate(train_data):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t
        data = batch.data[0].as_in_context(ctx)
        noise = nd.random_normal(shape=(batch_size, 2), ctx=ctx)

        with autograd.record():
            real_output = netD(data)
            errD_real = loss(real_output, real_label)
            
            fake = netG(noise)
            fake_output = netD(fake.detach())
            errD_fake = loss(fake_output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()

        trainerD.step(batch_size)
        metric.update([real_label,], [real_output,])
        metric.update([fake_label,], [fake_output,])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            output = netD(fake)
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch_size)

    name, acc = metric.get()
    metric.reset()
    print('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    print('time: %f' % (time.time() - tic))
    noise = nd.random_normal(shape=(100, 2), ctx=ctx)
    fake = netG(noise)
    plt.scatter(X[:,0].asnumpy(), X[:,1].asnumpy())
    plt.scatter(fake[:,0].asnumpy(), fake[:,1].asnumpy())
    plt.show()
```

## Checking the outcome

Let's now generate some fake data and check whether it looks real.

```python
noise = mx.nd.random_normal(shape=(100, 2), ctx=ctx)
fake = netG(noise)

plt.scatter(X[:,0].asnumpy(), X[:,1].asnumpy())
plt.scatter(fake[:,0].asnumpy(), fake[:,1].asnumpy())
plt.show()
```

## Conclusion 

A word of caution here - to get this to converge properly, we needed to adjust the learning rates *very carefully*. And for Gaussians, the result is rather mediocre - a simple mean and covariance estimator would have worked *much better*. However, whenever we don't have a really good idea of what the distribution should be, this is a very good way of faking it to the best of our abilities. Note that a lot depends on the power of the discriminating network. If it is weak, the fake can be very different from the truth. E.g. in our case it had trouble picking up anything along the axis of reduced variance. 
In summary, this isn't exactly easy to set and forget. One nice resource for dirty practioner's knowledge is [Soumith Chintala's handy list of tricks](https://github.com/soumith/ganhacks) for how to babysit GANs. 

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
