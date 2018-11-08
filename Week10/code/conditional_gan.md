# Conditional Generative Adversarial Networks

In [deep convolutional generative adversarial networks (DCGAN)](./dcgan.ipynb), 
we introduced how to generate human face images from random noise. With a DCGAN, we can generate images from random vectors, but what kind of images do we achieve? Can we specify it is a woman or man face?

In this notebook, we introduce [conditional GAN](https://arxiv.org/abs/1411.1784), which accepts label as part of the input for both generator and discriminator. With conditional GAN, you can choose which kind of data to generate with corresponding label. We'll train on MNIST dataset.

```python
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

import mxnet as mx
from mxnet import gluon, test_utils, autograd
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
```

## Set training parameters

```python
epochs = 1000
batch_size = 64
label_size = 10
latent_z_size = 100
hidden_units = 128
img_wd = 28
img_ht = 28

use_gpu = True
ctx = mx.gpu() if use_gpu else mx.cpu()

lr = 0.001
```

## Download and preprocess the MNIST Dataset

```python
# Pixel values of mnist image are normalized to be from 0 to 1.
mnist_data = test_utils.get_mnist()
train_data = mnist_data['train_data']
train_label = nd.one_hot(nd.array(mnist_data['train_label']), 10)
train_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size)
```

Visualize 4 images:

```python
def visualize(img_arr):
    plt.imshow((img_arr.asnumpy().reshape(img_wd, img_ht) * 255).astype(np.uint8), cmap='gray')
    plt.axis('off')

for i in range(4):
    plt.subplot(1,4,i+1)
    visualize(nd.array(train_data[i + 10]))
plt.show()
```

## Defining the networks

Input of generator is the concatenation of random noise vector and digit label one hot vector. It is followed by a fully connected layer and relu activation function. Output layer is composed by another fully connected layer and sigmoid activation function.

Similar to generator, input of discriminator is the concatenation of flatten image vector and digit label. It is followed by a fully connected layer and relu activation function. Output layer is composed by another fully connected layer and sigmoid activation function. In this tutorial we don't apply sigmoid function to output to make training process more numerically stable.

![](../img/cgan.png "Conditional GAN Architecture")

```python
w_init = mx.init.Xavier()

# Build the generator
netG = nn.HybridSequential()
with netG.name_scope():
    netG.add(nn.Dense(units=hidden_units, activation='relu', weight_initializer=w_init))
    netG.add(nn.Dense(units=img_wd * img_ht, activation='sigmoid', weight_initializer=w_init))

# Build the discriminator
netD = nn.HybridSequential()
with netD.name_scope():
    netD.add(nn.Dense(units=hidden_units, activation='relu', weight_initializer=w_init))
    netD.add(nn.Dense(units=1, weight_initializer=w_init))
```

## Setup Loss Function and Optimizer
We use binary cross-entropy as our loss function and use the Adam optimizer. We initialize the network's parameters by sampling from a normal distribution.

```python
# Loss
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# Initialize the generator and the discriminator
netG.initialize(ctx=ctx)
netD.initialize(ctx=ctx)

# Trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr})
```

## Training Loop
We recommend that you use a GPU for training this model. After a few epochs, we can see human-face-like images are generated.

```python
from datetime import datetime
import time
import logging

real_label = nd.ones((batch_size,), ctx=ctx)
fake_label = nd.zeros((batch_size,),ctx=ctx)

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()
metric = mx.metric.CustomMetric(facc)

stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
logging.basicConfig(level=logging.INFO)

for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    train_iter.reset()
    iter = 0
    for batch in train_iter:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, latent_z_size), ctx=ctx)
        D_input = nd.concat(data.reshape((data.shape[0], -1)), label)
        G_input = nd.concat(latent_z, label)

        with autograd.record():
            # train with real image
            output = netD(D_input)
            errD_real = loss(output, real_label)
            metric.update([real_label,], [output,])

            # train with fake image
            fake = netG(G_input)
            D_fake_input = nd.concat(fake.reshape((fake.shape[0], -1)), label)
            output = netD(D_fake_input.detach())
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label,], [output,])

        trainerD.step(batch.data[0].shape[0])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            fake = netG(G_input)
            D_fake_input = nd.concat(fake.reshape((fake.shape[0], -1)), label)
            output = netD(D_fake_input)
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch.data[0].shape[0])

        # Print log infomation every ten batches
        if iter % 10 == 0:
            name, acc = metric.get()
            logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
            logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d' 
                     %(nd.mean(errD).asscalar(), 
                       nd.mean(errG).asscalar(), acc, iter, epoch))
        iter = iter + 1
        btic = time.time()

    name, acc = metric.get()
    metric.reset()
    logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    logging.info('time: %f' % (time.time() - tic))

    # Visualize one generated image for each epoch
    fake_img = fake[0]
    visualize(fake_img)
    plt.show()
```

## Results
Given a trained generator, we can generate some images of digits.

```python
num_image = 4
for digit in range(10):
    for i in range(num_image):
        latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size), ctx=ctx)
        label = nd.one_hot(nd.array([[digit]]), 10).as_in_context(ctx)
        img = netG(nd.concat(latent_z, label.reshape((1, 10))))
        plt.subplot(10, 4, digit * 4 + i + 1)
        visualize(img[0])
plt.show()
```

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
