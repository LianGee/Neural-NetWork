
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>

# 数据集产生

原文参见<a href = "http://cs231n.github.io/neural-networks-case-study/">这里</a>


```python
import numpy as np
import matplotlib.pyplot as plt
N = 100  #numbers of points per class
D = 2  #dimesionality
K = 3  #number of classes
num_examples = N*K #number of examples
X = np.zeros((N*K, D))  #data matrix(each row = single example)
y = np.zeros(N*K, dtype=np.int8)  #classs labels
for j in xrange(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()
```

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>

前两行导入numpy和matplotlib两个库。numpy是科学计算需要用到的包，matplotlib是画图的包。  
这里产生K类数据，以（0，0）为圆心。  
每一类有N个数据，数据维度D = 2也就是点。  
X是一个[300，2] 的矩阵，也就是300个点  
y是数据标签，表征数据属于哪一类  
ix 的范围从N\*j 到 N \* j + N  
r是半径，用linspace产生从0.0到1之间步长为1/N的N个数据  
t是夹角，每一类有N个数据，因此我们需要产生N个夹角。random.randn(N)产生N个随机数保证每一类数据的夹角在linspace生成的不断增加的夹角的基础上有适当的波动，否则数据会呈线性。  
接下来给X赋值


```python
np.c_[np.array([[1,2,3]]), 0 , 0, np.array([[4,5,6]])]
```




    array([[1, 2, 3, 0, 0, 4, 5, 6]])



<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
np.c\_ 的作用就是实际上就是将其中的参数对象沿着第二轴链接成一个  
X[ix]注意到这里ix实际上是一个list，这种访问方式相当于for循环访问每一行  
y[ix]经过三次循环每一类中的N个数据都被附上同一个标签

<img src="http://i4.tietuku.com/87994e3dd74ad2ce.png">

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>

# 神经网络如何预测  
<img src = "https://github.com/LianGee/Neural-NetWork/blob/master/images/1.png">
神经网络使用前向传播进行预测。前向传播只不过是一对矩阵相乘并使用激活函数（softmax sigmoid tanh etc.）。  
假如x是该网络的2维输入，我们将按照如下计算预测值（也是二维的）：  
$z_1 = xW_1 + b_1$  
$a_1 = tanh(z_1)$  
$z_2 = a_1W_2 + b_2$ 
$a_2 = y = softmax(z_2)$  
$z_i$ 是输入层、$a_i$是输出层。W1,b1,W2,b2是需要从训练数据中学习的网络参数。你可以把它们看作是神经网络各层之间数据转换矩阵。  
<img src = "https://github.com/LianGee/Neural-NetWork/blob/master/images/2.png">  
上面的过程中我们发现，实际上是有两层的神经网络的，也就是说$z_1$作为第一层的输入，第一层输出$a_1$，然后将$a_1$代入线性方程作为第二层的输入，输出$a_2$， 然后将$a_2$作为softmax的输入。

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
## 初始化参数


```python
#initialize parameters randomly
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))
```

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>  
我们的数据集X是300\*2的矩阵。W是2\*3的矩阵，b是1\*3的矩阵。

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
## 目标函数  
$f(x;W) = W*x + b$


```python
# compute class scores for a linear classifier
scores = np.dot(X, W) + b
```

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>  
这个例子中我们有300个二维的点，相乘之后scores就是300\*2 \* 2\*3  = 300\*3 的矩阵。每一行的三个数据和三个类别对应。  
直观上来说，我们希望正确类别的数值要比其他类别要高。先来看以下是如何计算属于每一个类别的概率的  
softmax 函数：  
$p(y^i = j|x^i; \theta) = \dfrac{e^{f_j}}{\sum_{j = 1}^{K}e^{f_j}}$  
$L_i = -log(\dfrac{e^{f_{y_i}}}{\sum_{j}e^{f_j}})$   
这里f表示神经网络最后一层的输出（300 \* 3的矩阵的某一行），$y_i$  表示该数据所属的正确类别，再通过归一化取得正确类别的概率。  
当正确类别的概率非常小（趋近于0）的时候，损失函数的值趋近于正无穷。概率趋近于1的时候，值趋近于0。  
所以，正确类别的概率越大，$L_i$的值越小。  
因此，加上正规项的损失，最终的目标函数可以定义为：  
$$L = \underbrace{\dfrac{1}{N}\sum_{i}L_i}_{data \space loss} + \underbrace{\dfrac{1}{2}\lambda\sum_k\sum_lW_{k,l}^{2}}_{regularization \space loss}$$

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>  
# 计算损失函数 


```python
# get unnormalized probabilities
exp_scores = np.exp(scores)
# normalize them for each example
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
```

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>  
exp_scores 将scores（300\*3）所有数据取指数  
axis = 1按行加，keepdims保持最后结果为列向量。exp_scores 除去按行相加的和得到softmax函数的值。  


```python
corect_logprobs = -np.log(probs[range(num_examples),y]) 
print corect_logprobs.shape
```

    (300,)
    

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>

#  Softmax 线性分类器

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
现在我们再回过头来看之前的目标函数。  
如果你之前听说过两个类别的逻辑回归分类器（使用 sigmoid 函数），可以将Softmax分类器看做是一个多类别的逻辑回归分类器（使用softmax 函数）。  
softmax函数$f(x_i, W)$返回一个非常直观的结果即：属于该类别的概率。

softmax 函数：  
$$P_k(z) = \dfrac{e^{z_j}}{\sum_{k}e^{z_k}}$$  
$$L_i = -log(\dfrac{e^{f_{y_i}}}{\sum_{j}e^{f_j}})$$  
其中 $p_k$表示第probs每一行3类标签中正确标签对应的概率  
交叉熵矩阵corect_logprobs 是一个1\*300的矩阵。  
probs[range(num_examples), y] range(num_examples)产生0-299的序列，配合y，取得probs中所有正确标签的概率。  
然后对正确的概率求交叉熵。总的损失应该是交叉熵总的平均值和正规损失的和。因此损失函数:  
$$loss = \dfrac{1}{N}\sum_{i}L_i + \lambda R(W)$$  
展开就是上面提到的公式：  
$$ L = \dfrac{1}{N}\sum_iL_i + \lambda\sum_k\sum_lW_{k,l}^2$$  
(还有支持向量机的做法 <a href = "http://cs231n.github.io/linear-classify/">这里</a>)

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
那么，接下来的任务就是求得使得这个目标函数最小的参数$W, b$了。(注意到$L_i$与f相关，而$f = Wx_i + b$)  
通常，在线性回归中，使用sigmoid进行分类的时候，我们的做法是用梯度下降直接求最大似然估计的最大值。但是注意到那时，我们的参数只有1个$\theta$。  
现在这里需要求出的参数是两个$W, b$。 不难看出，$L_i$是从概率$f_j(z)$计算出来的，而概率依赖于f。  
$L_i = -log$

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
## 学习参数  



```python

```
