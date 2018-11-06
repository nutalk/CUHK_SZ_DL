# 门控循环单元（GRU）

上一节介绍了循环神经网络中的梯度计算方法。我们发现，循环神经网络的梯度由于有累乘，它可能会衰减或爆炸。虽然裁剪梯度可以应对梯度爆炸，但无法解决梯度衰减的问题。通常由于这个原因，循环神经网络在实际中较难捕捉时间步距离较大的样本之间的依赖关系，因为信息传递时容易衰减或爆炸。

门控循环神经网络（gated recurrent neural network）的提出，是为了更好地捕捉时间序列中时间步距离较大的依赖关系。它们通过可以学习的门来控制信息的流动，使得信息要么简单通过或者要么被阻挡。其中，门控循环单元（gated recurrent unit，简称GRU）是一种常用的门控循环神经网络 [1, 2]。另一种常见门控循环神经网络，长短期记忆，则将在下一节介绍。


## 门控循环单元

下面将介绍门控循环单元的设计。它引入了重置门和更新门的概念，从而修改了循环神经网络中隐藏状态的计算方式。

### 重置门和更新门

如图6.4所示，门控循环单元中的重置门（reset gate）和更新门（update gate）均由输入为当前时间步输入$\boldsymbol{X}_t$与上一时间步隐藏状态$\boldsymbol{H}_{t-1}$，且激活函数为sigmoid函数的全连接层计算得出。


![门控循环单元中重置门和更新门的计算。](./gru_1.svg)


具体来说，假设隐藏单元个数为$h$，给定时间步$t$的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$）和上一时间步隐藏状态$\boldsymbol{H}_{t-1} \in \mathbb{R}^{n \times h}$。重置门$\boldsymbol{R}_t \in \mathbb{R}^{n \times h}$和更新门$\boldsymbol{Z}_t \in \mathbb{R}^{n \times h}$的计算如下：

$$
\begin{aligned}
\boldsymbol{R}_t = \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xr} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hr} + \boldsymbol{b}_r),\\
\boldsymbol{Z}_t = \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xz} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hz} + \boldsymbol{b}_z),
\end{aligned}
$$

其中$\boldsymbol{W}_{xr}, \boldsymbol{W}_{xz} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hr}, \boldsymbol{W}_{hz} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_r, \boldsymbol{b}_z \in \mathbb{R}^{1 \times h}$是偏移参数。激活函数$\sigma$是sigmoid函数。[“多层感知机”](../chapter_deep-learning-basics/mlp.md)一节中介绍过，sigmoid函数可以将元素的值变换到0和1之间。因此，重置门$\boldsymbol{R}_t$和更新门$\boldsymbol{Z}_t$中每个元素的值域都是$[0, 1]$。

### 候选隐藏状态

接下来，门控循环单元将计算候选隐藏状态来辅助稍后的隐藏状态计算。如图6.5所示，我们将当前时间步重置门的输出与上一时间步隐藏状态做按元素乘法。如果重置门中元素值接近0，那么意味着重置对应隐藏状态元素为0，即丢弃（或者说重置）来自上一时间步的历史信息。如果元素值接近1，那么表示通过来自上一时间步的信息。

然后，将重置后的上一时间步的隐藏状态与当前时间步的输入合并，再通过含tanh激活函数的全连接层计算出候选隐藏状态，其所有元素的值域为$[-1, 1]$。

![门控循环单元中候选隐藏状态的计算。这里的乘号是按元素乘法。](./gru_2.svg)

具体来说，时间步$t$的候选隐藏状态$\tilde{\boldsymbol{H}}_t \in \mathbb{R}^{n \times h}$的计算为

$$\tilde{\boldsymbol{H}}_t = \text{tanh}(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \left(\boldsymbol{R}_t \odot \boldsymbol{H}_{t-1}\right) \boldsymbol{W}_{hh} + \boldsymbol{b}_h),$$

其中$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$是偏移参数。从上面这个公式可以看出，重置门控制了包含时间序列历史信息的上一个时间步的隐藏状态如何流入当前时间步的候选隐藏状态，它可以用来丢弃与预测未来无关的历史信息。

### 隐藏状态

最后，时间步$t$的隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times h}$的计算使用当前时间步的更新门$\boldsymbol{Z}_t$来对上一时间步的隐藏状态$\boldsymbol{H}_{t-1}$和当前时间步的候选隐藏状态$\tilde{\boldsymbol{H}}_t$做组合：

$$\boldsymbol{H}_t = \boldsymbol{Z}_t \odot \boldsymbol{H}_{t-1}  + (1 - \boldsymbol{Z}_t) \odot \tilde{\boldsymbol{H}}_t.$$


![门控循环单元中隐藏状态的计算。这里的乘号是按元素乘法。](./gru_3.svg)


值得注意的是，更新门可以控制隐藏状态应该如何被包含当前时间步信息的候选隐藏状态所更新，如图6.6所示。假设更新门在时间步$t'$到$t$（$t' < t$）之间一直近似1。那么，在时间步$t'$到$t$之间的输入信息几乎没有流入时间步$t$的隐藏状态$\boldsymbol{H}_t$。实际上，这可以看作是较早时刻的隐藏状态$\boldsymbol{H}_{t'-1}$一直通过时间保存并传递至当前时间步$t$。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

我们对门控循环单元的设计稍作总结：

* 重置门有助于捕捉时间序列里短期的依赖关系。
* 更新门有助于捕捉时间序列里长期的依赖关系。

## 载入数据集

为了实现并展示门控循环单元，我们依然使用周杰伦歌词数据集来训练模型作词。这里除门控循环单元以外的实现已在[“循环神经网络”](rnn.md)一节中介绍。我们先导入包和模块，并读取数据集。

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import nd
from mxnet.gluon import rnn

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = gb.load_data_jay_lyrics()
```

## GRU的从零开始实现


### 初始化模型参数

以下部分对模型参数进行初始化。超参数`num_hiddens`定义了隐藏单元的个数。

```{.python .input  n=2}
num_inputs = vocab_size
num_hiddens = 256
num_outputs = vocab_size
ctx = gb.try_gpu()

def get_params():
    _one = lambda shape: nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    _three = lambda : (_one((num_inputs, num_hiddens)), 
                       _one((num_hiddens, num_hiddens)), 
                       nd.zeros(num_hiddens, ctx=ctx))    
    W_xz, W_hz, b_z = _three()  # 更新门参数。    
    W_xr, W_hr, b_r = _three()  # 重置门参数。    
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数。
    # 输出层参数。
    W_hy = _one((num_hiddens, num_outputs))
    b_y = nd.zeros(num_outputs, ctx=ctx)
    # 创建梯度。
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params
```

### 定义模型

首先定义隐藏状态初始化函数，同`init_rnn_state`一样它返回由一个形状为（`batch_size`，`num_hiddens`）的值为0的NDArray组成的列表。

```{.python .input  n=3}
def init_gru_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
```

下面根据门控循环单元的计算表达式定义模型。

```{.python .input  n=4}
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hy, b_y = params
    H, = state
    outputs = []
    for X in inputs:        
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)
        H_tilda = nd.tanh(nd.dot(X, W_xh) + R * nd.dot(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return outputs, (H,)
```

### 训练模型并创作歌词

使用同前一节类似的超参数训练，但我们这里减少了迭代周期数，且训练模型时只采用了相邻采样。

```{.python .input  n=5}
num_epochs = 160
num_steps = 35
batch_size = 32
lr = 1e2
clipping_theta = 1e-2
prefixes = ['分开', '不分开']
pred_period = 40
pred_len = 50
```

设置好超参数后，我们将训练模型并跟据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词。我们每过30个迭代周期便根据当前训练的模型创作一段歌词。。

```{.python .input}
gb.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                         vocab_size, ctx, corpus_indices, idx_to_char,
                         char_to_idx, False, num_epochs, num_steps, lr,
                         clipping_theta, batch_size, pred_period, pred_len,
                         prefixes)
```

## GRU的Gluon实现

在Gluon中我们直接调用rnn模块中的GRU类即可。

```{.python .input  n=6}
gru_layer = rnn.GRU(num_hiddens)
model = gb.RNNModel(gru_layer, vocab_size)

gb.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                               corpus_indices, idx_to_char, char_to_idx,
                               num_epochs, num_steps, lr, clipping_theta,
                               batch_size, pred_period, pred_len, prefixes)
```

## 小结

* 门控循环神经网络可以更好地捕捉时间序列中时间步距离较大的依赖关系，它包括门控循环单元和长短期记忆。
* 门控循环单元引入了门的概念，从而修改了循环神经网络中隐藏状态的计算方式。它包括重置门、更新门、候选隐藏状态和隐藏状态。
* 重置门有助于捕捉时间序列里短期的依赖关系。
* 更新门有助于捕捉时间序列里长期的依赖关系。


## 练习

* 假设时间步$t' < t$。如果我们只希望用时间步$t'$的输入来预测时间步$t$的输出，每个时间步的重置门和更新门的值最好是多少？
* 调调超参数，观察并分析对运行时间、困惑度以及创作歌词的结果造成的影响。
* 在相同条件下，比较门控循环单元和循环神经网络的运行时间。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4042)

![](../img/qr_gru.svg)

## 参考文献

[1] Cho, K., Van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014). On the properties of neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259.

[2] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
