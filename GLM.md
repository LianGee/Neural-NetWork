
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
[1.总览指数簇分布](#0)  
[2.伯努利分布的指数簇形式推导](#1)  
[3.高斯分布的指数簇形式推导](#2)  
[4.一般线性模型指数簇形式推导](#3)  
　　[4.1 普通最小二乘指数簇形式推导](#3.1)  
　　[4.2 logistic 回归指数簇形式推导](#3.2)  
　　[4.3 softmax 回归指数簇形式推导](#3.3)  
</font>
</div>

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h1 id = "0">指数簇分布</h1>
在做分类的时候，我们使用过sigmoid，softmax函数。之所以使用这些函数是有一套理论支持的。
首先引入指数簇的概念数学表达式如下：
$$p(y;n) = b(y)exp(\eta^T(y) - a(\eta))$$
许多的分布都可以变型为这种分布表达形式。<p>
其中，$\eta$被称之为自然参数(natural parameter)或者典范参数(canonical parameter)<p>
$T(y)$是统计分量(sufficient static)(对于我们考虑的分布来说，通常$T(y) = y$)<p>
$a(\eta)$是(log partition function), $e^{-a(\eta)}$是一个规范化常数，使得分布的和为1<p>
给定T, a, b,通过改变参数 $\eta$ 就可以得到不同的分布。
</font>
</div>

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h1 id = "1">伯努利分布</h1>
如果一个数据集需要分成两类，我们通常使用sigmoid函数。通常的做法是：<p>
1.通过线性拟合得到$h(\theta) = \theta_{1}x_1 + \theta_{2}x_2 + \dot + \theta_{n}x_n$<p>
2.代入sigmoid函数得到 $H(x) = sigmoid(\theta^TX)$，这样输出就只有0和1<p>
3.拟合出参数使得输出结果的概率最大,最大似然函数$\ell(\theta) = ln(L(\theta)) = \sum_{i = 0}^{m}y_iln(h_\theta(X^i)) + \sum_{i = 0}^{m}(1 - y_i)ln(1 - h_\theta(X^i))$<p>
第三步，求$\ell(\theta)$的最大值，可以采用牛顿方法，梯度上升等<p>
使用梯度上升方法得到的$\theta$ 的更新公式：$$\theta_j := \theta_j + \sum_{i = 0}^{m}(y^i - h_\theta(X^i))X_j^i$$

事实上，如果将伯努利分布写成指数簇分布形式：<p>
<table border = "1" width="100">
<tr>
    <th>0</th>
    <th>1</th>
</tr>
<tr>
    <th>$1 - \phi$</th>
    <th>$\phi$</th>
</tr>
</table>
<p align="left">
<font size = "5", color = "black">
$p(y;\phi) = \phi^y(1 - \phi)^{(1- y)}$<p>
$= e^{(yln\phi + (1- y)ln(1 - \phi))}$<p>
$= e^{((ln(\phi) - ln(1 - \phi))y + ln(1 - \phi))}$<p>
$= e^{(ln(\dfrac{\phi}{1 - \phi})y + ln(1 - \phi))}$<p>
</font>
因此，令$\eta = ln(\phi/(1 - \phi))$ (有趣的是其反函数是：$\phi = 1/(1 + e^{-\eta})$),并且<p>
$T(y) = y$<p>
$a(\eta) = -ln(1 - \phi) = ln(1 + e^\eta)$<p>
$b(y) = 1$<p>
</p>
</font>
</div>

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h1 id = "2">高斯分布</h1>
推导类似伯努利分布得到的结果：
$\eta = \mu$<p>
$T(y) = y$<p>
$a(\eta) = -\mu^2/2 = \eta^2/2$<p>
$b(y) = (1/\sqrt{2\pi})e^{-y^2/2}$<p>
</font>
</div>

## <h1 id = "3">GLMS</h1>
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
为了推导出GLM，做出三个假设（不用担心假设不成立，至少对于logistic回归和softmax回归是合理的）  
1. $y|x;\theta ～ ExponentialFamily(\eta)$  
2. 给定x，我们的目标是预测 T(y) 的预期值。在大部分例子中，都有T(y) = y， 因此意味着我们通过学习得到的假设满足$h(x) = E(y|x)$  
3. 自然参数和输入变量是线性相关的，也就是说 $\eta = \theta^Tx$(如果自然参数是向量，则 $\eta_i = \theta_i^Tx$)
</font>
</div>

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h2 id = "3.1">普通的最小二乘</h2>  

为了说明普通的最小二乘是GLM的特例，设定目标变量y是连续的，且服从高斯分布即$Y ～　N(\mu, \sigma^2)$  
通过高斯分布变型为指数簇的结果,我们有：  
$h_\theta(x) = E(y|x;\theta)$
$= \mu$  
$= \eta$  
$= \theta^Tx$
</font>
</div>

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h2 id = "3.2">logistic 回归</h2>   

逻辑回归是二元分类的，因此很容易想到伯努利分布，所以预测结果的期望和伯努利分布中为1的概率是相等的即：  
$h_\theta(x) = E(y|x;\theta)$  
$= \phi$  
$= 1/(1 + e^{-\phi})$  
$= 1/(1 + e^{\theta^Tx})$
</font>
</div>

<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>

<h2 id = "3.3">softmax 回归</h2>   
考虑分类结果不只两种的情况，需要采用多项式分布  
<table>
    <tr>
        <th>1</th><th>2</th><th>$\dots$</th><th>k-1</th><th>k</th>
    </tr>
    <tr>
        <th>$\phi_1$</th><th>$\phi_2$<th>$\dots$</th><th>$\phi_{k-1}$</th><th>$\phi_{k}$</th>
    </tr>
</table>

为了参数化多项式分布的k个可能结果，有人可能会用k个参数来说明每一种情况的可能性，但是这些参数是冗余的，并且不是独立的。  
实际上知道任何其中的k-1个，剩下的一个就可以求出，因为概率满足 $\sum_{i = 1}^{k}\phi_i = 1$，因此我们用k-1个参数<p>
$$\phi_1, \phi_2, \dots, \phi_{k-1}$$  
对多项式分布进行参数化<p>
$\phi_i = p(y = i; \phi), and \space p(y = k; \phi) = 1 - \sum_{i = 1}^{k - 1}\phi_i$  
定义$T(y) \in R^{k - 1}$  
<p>
$T(1) = \begin{bmatrix}
1\\
0\\
\vdots \\
0
\end{bmatrix}$,
$T(2) = \begin{bmatrix}
0\\
1\\
\vdots \\
0
\end{bmatrix}$,
$\dots$,
$T(k - 1) = \begin{bmatrix}
0\\
0\\
\vdots \\
1
\end{bmatrix}$,
$T(k) = \begin{bmatrix}
0\\
0\\
\vdots \\
0
\end{bmatrix}$  
<p>
引入一个记号方便书写：  
1{True} = 1, 1{False} = 0, 例如1{2 = 3} = 0， 1{3 = 5- 2} = 1  
因此，T(y) 和 y 的关系可以表示为： $(T(y)_i = 1\{y = i\})$  
那么：<p>  
$E(T(y)_i) = p(y = i) = \phi_i$ 因此： <p> 
$p(y;\phi) = \phi_1^{1\{y = 1\}}\phi_2^{1\{y = 2\}}\dots\phi_k^{1\{y = k\}}$<p>  
$= \phi_1^{1\{y = 1\}}\phi_2^{1\{y = 2\}}\dots\phi_k^{1 - \sum_{i = 1}^{k}1\{y = i\}}$<p>  
$= \phi_1^{T(y)_1}\phi_2^{T(y)_2}\dots\phi_k^{1 - \sum_{i = 1}^{k}T(y)_i}$<p>  
$= exp(T(y)_{1}ln\phi_1 + T(y)_{2}ln\phi_2 + \dots + (1 - \sum_{i = 1}^{k}T(y)_i)ln\phi_k) $<p>  
$= exp(T(y)_{1}ln(\phi_1/\phi_k) + T(y)_{2}ln(\phi_2/\phi_k) + \dots + T(y)_{k-1}ln(\phi_{k - 1}/\phi_k) + ln\phi_k)$<p>  
$= b(y)exp(\eta^TT(y) - a(\eta))$  
其中：  
$b(y) = 1$<p>  
$\eta = \begin{bmatrix}
ln(\phi_1/\phi_k)\\
ln(\phi_2/\phi_k)\\
\dots \\
ln(\phi_k-1/\phi_k)
\end{bmatrix}$<p>  
$a(\eta) = -ln(\phi_k)$<p>  
<p>链接函数$\eta_i = \dfrac{\phi_i}{\phi_k}$，为了方便，定义$\eta_i = ln(\dfrac{\phi_k}{\phi_k}) = 0$  
可得：<p>  
$e^{\eta_i} = \dfrac{\phi_i}{\phi_k}$
$\phi_k e^{\phi_i} = \phi_i$
$\phi_k\sum_{i = 1}^{k}e^{\eta_i} = \sum_{i = 1}^{k}\phi_i = 1$<p>  
因此 $\phi_k = 1/\sum_{i = 1}^{k}e^{\eta_i}$，反代回去得到响应函数：<p>  
$\eta_i = \dfrac{e^{\eta_i}}{\sum_{j = 1}^{k}e^{\eta_j} }$<p>  
从$\eta_i = \theta_i^Tx(for i = 1,\dots,k-1)$得到：<p>  
$p(y = i|x;\theta) = \phi_i$
$= \dfrac{e^{\theta_i^Tx}}{\sum_{j = 1}^{k}e^{\eta_j}}$
$= \dfrac{e^{\eta_i}}{\sum_{j = 1}^{k}e^{\theta_j^Tx}}$  
从$\eta$到$\phi$的映射叫做softmax函数。<p>  
这个应用于二元分类的推广，当$y \in\{1, \dots, k\}$<p>  
$h_\theta(x) = E(T(y)|x; \theta)$
$= \begin{bmatrix}
\phi_1\\
\phi_2\\
\vdots\\
\phi_k\\
\end{bmatrix}$<p>  
与最小二乘和逻辑回归类似类似：<p>  
$\ell(\theta) = \sum_{i = 1}^{m}lnp(y^{(i)|x^{(i)};\theta})$
$= \sum_{i = 1}^{m}ln\prod_{w = 1}^{k}(\dfrac{e^{\eta_ix^{(i)}}}{\sum_{j = 1}^{k}e^{\eta_jx^{(i)}} })^{1\{y^{(i)} = w\}}$<p>
</p>
</font>
</div>