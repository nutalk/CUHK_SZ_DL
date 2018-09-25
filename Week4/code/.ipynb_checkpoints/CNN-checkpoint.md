# Convolutional neural networks from scratch

Now let's take a look at *convolutional neural networks* (CNNs), the models people really use for classifying images.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
ctx = mx.cpu()
# ctx = mx.gpu()
mx.random.seed(1)
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zli/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n  from ._conv import register_converters as _register_converters\n"
 }
]
```

## MNIST data (last one, we promise!)

```{.python .input}
batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## Convolutional neural networks (CNNs)

In the [previous example](5a-mlp-scratch.ipynb), we connected the nodes of our neural networks in what seems like the simplest possible way. Every node in each layer was connected to every node in the subsequent layers. 

![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/multilayer-perceptron.png?raw=true)

This can require a lot of parameters! If our input were a 256x256 color image (still quite small for a photograph), and our network had 1,000 nodes in the first hidden layer, then our first weight matrix would require (256x256x3)x1000 parameters. That's nearly 200 million. Moreover the hidden layer would ignore all the spatial structure in the input image even though we know the local structure represents a powerful source of prior knowledge. 

Convolutional neural networks incorporate convolutional layers. These layers associate each of their nodes with a small window, called a *receptive field*, in the previous layer, instead of connecting to the full layer. This allows us to first learn local features via transformations that are applied in the same way for the top right corner as for the bottom left. Then we collect all this local information to predict global qualities of the image (like whether or not it depicts a dog). 

![](http://cs231n.github.io/assets/cnn/depthcol.jpeg)
(Image credit: Stanford cs231n http://cs231n.github.io/assets/cnn/depthcol.jpeg)

In short, there are two new concepts you need to grok here. First, we'll be introducting *convolutional* layers. Second, we'll be interleaving them with *pooling* layers. 

##  Parameters

Each node in a convolutional layer is associated with a 3D block (height x width x channel) in the input tensor. Moreover, the convolutional layer itself has multiple output channels. So the layer is parameterized by a 4 dimensional weight tensor, commonly called a *convolutional kernel*. 

The output tensor is produced by sliding the kernel across the input image skipping locations according to a pre-defined *stride* (but we'll just assume that to be 1 in this tutorial). Let's initialize some such kernels from scratch.

```{.python .input}
#######################
#  Set the scale for weight initialization and choose 
#  the number of hidden units in the fully-connected layer 
####################### 
weight_scale = .01
num_fc = 128
num_filter_conv_layer1 = 20
num_filter_conv_layer2 = 50

W1 = nd.random_normal(shape=(num_filter_conv_layer1, 1, 3,3), scale=weight_scale, ctx=ctx) 
b1 = nd.random_normal(shape=num_filter_conv_layer1, scale=weight_scale, ctx=ctx)

W2 = nd.random_normal(shape=(num_filter_conv_layer2, num_filter_conv_layer1, 5, 5),
                                                    scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=num_filter_conv_layer2, scale=weight_scale, ctx=ctx)

W3 = nd.random_normal(shape=(800, num_fc), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
b4 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]
```

And assign space for gradients

```{.python .input}
for param in params:
    param.attach_grad()
```

## Convolving with MXNet's NDArrray

To write a convolution when using *raw MXNet*, we use the function ``nd.Convolution()``. This function takes a few important arguments: inputs (``data``), a 4D weight matrix (``weight``), a bias (``bias``), the shape of the kernel (``kernel``), and a number of filters (``num_filter``).

```{.python .input}
for data, _ in train_data:
    data = data.as_in_context(ctx)
    break
conv = nd.Convolution(data=data, weight=W1, bias=b1, kernel=(3,3), num_filter=num_filter_conv_layer1)
print(conv.shape)
```

Note the shape. The number of examples (64) remains unchanged. The number of channels (also called *filters*) has increased to 20. And because the (3,3) kernel can only be applied in 26 different heights and widths (without the kernel busting over the image border), our output is 26,26. There are some weird padding tricks we can use when we want the input and output to have the same height and width dimensions, but we won't get into that now.

## Average pooling

The other new component of this model is the pooling layer. Pooling gives us a way to downsample in the spatial dimensions. Early convnets typically used average pooling, but max pooling tends to give better results.

```{.python .input}
pool = nd.Pooling(data=conv, pool_type="max", kernel=(2,2), stride=(2,2))
print(pool.shape)
```

Note that the batch and channel components of the shape are unchanged but that the height and width have been downsampled from (26,26) to (13,13).

## Activation function

```{.python .input}
def relu(X):
    return nd.maximum(X,nd.zeros_like(X))
```

## Softmax output

```{.python .input}
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition
```

## Softmax cross-entropy loss

```{.python .input}
def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)
```

## Define the model

Now we're ready to define our model

```{.python .input}
def net(X, debug=False):
    ########################
    #  Define the computation of the first convolutional layer
    ########################
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3),
                                  num_filter=num_filter_conv_layer1)
    h1_activation = relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))
        
    ########################
    #  Define the computation of the second convolutional layer
    ########################
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5,5),
                                  num_filter=num_filter_conv_layer2)
    h2_activation = relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))
    
    ########################
    #  Flattening h2 so that we can feed it into a fully-connected layer
    ########################
    h2 = nd.flatten(h2)
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))
    
    ########################
    #  Define the computation of the third (fully-connected) layer
    ########################
    h3_linear = nd.dot(h2, W3) + b3
    h3 = relu(h3_linear)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))
        
    ########################
    #  Define the computation of the output layer
    ########################
    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
    
    return yhat_linear

```

## Test run

We can now print out the shapes of the activations at each layer by using the debug flag.

```{.python .input}
output = net(data, debug=True)
```

## Optimizer

```{.python .input}
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Evaluation metric

```{.python .input}
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
```

## The training loop

```{.python .input}
epochs = 1
learning_rate = .01
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, num_outputs)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
 
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
```

## Conclusion

Contained in this example are nearly all the important ideas you'll need to start attacking problems in computer vision. While state-of-the-art vision systems incorporate a few more bells and whistles, they're all built on this foundation. Believe it or not, if you knew just the content in this tutorial 5 years ago, you could probably have sold a startup to a Fortune 500 company for millions of dollars. Fortunately (or unfortunately?), the world has gotten marginally more sophisticated, so we'll have to come up with some more sophisticated tutorials to follow.

## Next
[Convolutional neural networks with gluon](../chapter04_convolutional-neural-networks/cnn-gluon.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
