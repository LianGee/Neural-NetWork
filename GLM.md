<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
**[1.����ָ���طֲ�](#0)**  
**[2.��Ŭ���ֲ���ָ������ʽ�Ƶ�](#1)**  
**[3.��˹�ֲ���ָ������ʽ�Ƶ�](#2)**  
**[4.һ������ģ��ָ������ʽ�Ƶ�](#3)**  
����**[4.1 ��ͨ��С����ָ������ʽ�Ƶ�](#3.1)**  
����**[4.2 logistic �ع�ָ������ʽ�Ƶ�](#3.2)**  
����**[4.3 softmax �ع�ָ������ʽ�Ƶ�](#3.3)**  
</font>
</div>
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h1 id = "0">ָ���طֲ�</h1>
���������ʱ������ʹ�ù�sigmoid��softmax������֮����ʹ����Щ��������һ������֧�ֵġ�
��������ָ���صĸ�����ѧ���ʽ���£�
$$p(y;n) = b(y)exp(\eta^T(y) - a(\eta))$$
���ķֲ������Ա���Ϊ���ֲַ������ʽ��<p>
���У�$\eta$����֮Ϊ��Ȼ����(natural parameter)���ߵ䷶����(canonical parameter)<p>
$T(y)$��ͳ�Ʒ���(sufficient static)(�������ǿ��ǵķֲ���˵��ͨ��$T(y) = y$)<p>
$a(\eta)$��(log partition function), $e^{-a(\eta)}$��һ���淶��������ʹ�÷ֲ��ĺ�Ϊ1<p>
����T, a, b,ͨ���ı���� $\eta$ �Ϳ��Եõ���ͬ�ķֲ���
</font>
</div>
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h1 id = "1">��Ŭ���ֲ�</h1>
���һ�����ݼ���Ҫ�ֳ����࣬����ͨ��ʹ��sigmoid������ͨ���������ǣ�<p>
1.ͨ��������ϵõ�$h(\theta) = \theta_{1}x_1 + \theta_{2}x_2 + \dot + \theta_{n}x_n$<p>
2.����sigmoid�����õ� $H(x) = sigmoid(\theta^TX)$�����������ֻ��0��1<p>
3.��ϳ�����ʹ���������ĸ������,�����Ȼ����$\ell(\theta) = ln(L(\theta)) = \sum_{i = 0}^{m}y_iln(h_\theta(X^i)) + \sum_{i = 0}^{m}(1 - y_i)ln(1 - h_\theta(X^i))$<p>
����������$\ell(\theta)$�����ֵ�����Բ���ţ�ٷ������ݶ�������<p>
ʹ���ݶ����������õ���$\theta$ �ĸ��¹�ʽ��$$\theta_j := \theta_j + \sum_{i = 0}^{m}(y^i - h_\theta(X^i))X_j^i$$

��ʵ�ϣ��������Ŭ���ֲ�д��ָ���طֲ���ʽ��<p>
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
��ˣ���$\eta = ln(\phi/(1 - \phi))$ (��Ȥ�����䷴�����ǣ�$\phi = 1/(1 + e^{-\eta})$),����<p>
$T(y) = y$<p>
$a(\eta) = -ln(1 - \phi) = ln(1 + e^\eta)$<p>
$b(y) = 1$<p>
</p>
</font>
</div>
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h1 id = "2">��˹�ֲ�</h1>
�Ƶ����Ʋ�Ŭ���ֲ��õ��Ľ����
$\eta = \mu$<p>
$T(y) = y$<p>
$a(\eta) = -\mu^2/2 = \eta^2/2$<p>
$b(y) = (1/\sqrt{2\pi})e^{-y^2/2}$<p>
</font>
</div>
## <h1 id = "3">GLMS</h1>
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
Ϊ���Ƶ���GLM�������������裨���õ��ļ��費���������ٶ���logistic�ع��softmax�ع��Ǻ���ģ�  
1. $y|x;\theta �� ExponentialFamily(\eta)$  
2. ����x�����ǵ�Ŀ����Ԥ�� T(y) ��Ԥ��ֵ���ڴ󲿷������У�����T(y) = y�� �����ζ������ͨ��ѧϰ�õ��ļ�������$h(x) = E(y|x)$  
3. ��Ȼ���������������������صģ�Ҳ����˵ $\eta = \theta^Tx$(�����Ȼ�������������� $\eta_i = \theta_i^Tx$)
</font>
</div>
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h2 id = "3.1">��ͨ����С����</h2>  

Ϊ��˵����ͨ����С������GLM���������趨Ŀ�����y�������ģ��ҷ��Ӹ�˹�ֲ���$Y ����N(\mu, \sigma^2)$  
ͨ����˹�ֲ�����Ϊָ���صĽ��,�����У�  
$h_\theta(x) = E(y|x;\theta)$
$= \mu$  
$= \eta$  
$= \theta^Tx$
</font>
</div>
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>
<h2 id = "3.2">logistic �ع�</h2>   

�߼��ع��Ƕ�Ԫ����ģ���˺������뵽��Ŭ���ֲ�������Ԥ�����������Ͳ�Ŭ���ֲ���Ϊ1�ĸ�������ȵļ���  
$h_\theta(x) = E(y|x;\theta)$  
$= \phi$  
$= 1/(1 + e^{-\phi})$  
$= 1/(1 + e^{\theta^Tx})$
</font>
</div>
<div style="font-family:elephant, 'Microsoft YaHei';"> 
<font style="line-height:2;", size = '3'>

<h2 id = "3.3">softmax �ع�</h2>   
���Ƿ�������ֻ���ֵ��������Ҫ���ö���ʽ�ֲ�  
<table>
    <tr>
        <th>1</th><th>2</th><th>$\dots$</th><th>k-1</th><th>k</th>
    </tr>
    <tr>
        <th>$\phi_1$</th><th>$\phi_2$<th>$\dots$</th><th>$\phi_{k-1}$</th><th>$\phi_{k}$</th>
    </tr>
</table>

Ϊ�˲���������ʽ�ֲ���k�����ܽ�������˿��ܻ���k��������˵��ÿһ������Ŀ����ԣ�������Щ����������ģ����Ҳ��Ƕ����ġ�  
ʵ����֪���κ����е�k-1����ʣ�µ�һ���Ϳ����������Ϊ�������� $\sum_{i = 1}^{k}\phi_i = 1$�����������k-1������<p>
$$\phi_1, \phi_2, \dots, \phi_{k-1}$$  
�Զ���ʽ�ֲ����в�����<p>
$\phi_i = p(y = i; \phi), and \space p(y = k; \phi) = 1 - \sum_{i = 1}^{k - 1}\phi_i$  
����$T(y) \in R^{k - 1}$  
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
����һ���Ǻŷ�����д��  
1{True} = 1, 1{False} = 0, ����1{2 = 3} = 0�� 1{3 = 5- 2} = 1  
��ˣ�T(y) �� y �Ĺ�ϵ���Ա�ʾΪ�� $(T(y)_i = 1\{y = i\})$  
��ô��<p>  
$E(T(y)_i) = p(y = i) = \phi_i$ ��ˣ� <p> 
$p(y;\phi) = \phi_1^{1\{y = 1\}}\phi_2^{1\{y = 2\}}\dots\phi_k^{1\{y = k\}}$<p>  
$= \phi_1^{1\{y = 1\}}\phi_2^{1\{y = 2\}}\dots\phi_k^{1 - \sum_{i = 1}^{k}1\{y = i\}}$<p>  
$= \phi_1^{T(y)_1}\phi_2^{T(y)_2}\dots\phi_k^{1 - \sum_{i = 1}^{k}T(y)_i}$<p>  
$= exp(T(y)_{1}ln\phi_1 + T(y)_{2}ln\phi_2 + \dots + (1 - \sum_{i = 1}^{k}T(y)_i)ln\phi_k) $<p>  
$= exp(T(y)_{1}ln(\phi_1/\phi_k) + T(y)_{2}ln(\phi_2/\phi_k) + \dots + T(y)_{k-1}ln(\phi_{k - 1}/\phi_k) + ln\phi_k)$<p>  
$= b(y)exp(\eta^TT(y) - a(\eta))$  
���У�  
$b(y) = 1$<p>  
$\eta = \begin{bmatrix}
ln(\phi_1/\phi_k)\\
ln(\phi_2/\phi_k)\\
\dots \\
ln(\phi_k-1/\phi_k)
\end{bmatrix}$<p>  
$a(\eta) = -ln(\phi_k)$<p>  
<p>���Ӻ���$\eta_i = \dfrac{\phi_i}{\phi_k}$��Ϊ�˷��㣬����$\eta_i = ln(\dfrac{\phi_k}{\phi_k}) = 0$  
�ɵã�<p>  
$e^{\eta_i} = \dfrac{\phi_i}{\phi_k}$
$\phi_k e^{\phi_i} = \phi_i$
$\phi_k\sum_{i = 1}^{k}e^{\eta_i} = \sum_{i = 1}^{k}\phi_i = 1$<p>  
��� $\phi_k = 1/\sum_{i = 1}^{k}e^{\eta_i}$��������ȥ�õ���Ӧ������<p>  
$\eta_i = \dfrac{e^{\eta_i}}{\sum_{j = 1}^{k}e^{\eta_j} }$<p>  
��$\eta_i = \theta_i^Tx(for i = 1,\dots,k-1)$�õ���<p>  
$p(y = i|x;\theta) = \phi_i$
$= \dfrac{e^{\theta_i^Tx}}{\sum_{j = 1}^{k}e^{\eta_j}}$
$= \dfrac{e^{\eta_i}}{\sum_{j = 1}^{k}e^{\theta_j^Tx}}$  
��$\eta$��$\phi$��ӳ�����softmax������<p>  
���Ӧ���ڶ�Ԫ������ƹ㣬��$y \in\{1, \dots, k\}$<p>  
$h_\theta(x) = E(T(y)|x; \theta)$
$= \begin{bmatrix}
\phi_1\\
\phi_2\\
\vdots\\
\phi_k\\
\end{bmatrix}$<p>  
����С���˺��߼��ع��������ƣ�<p>  
$\ell(\theta) = \sum_{i = 1}^{m}lnp(y^{(i)|x^{(i)};\theta})$
$= \sum_{i = 1}^{m}ln\prod_{w = 1}^{k}(\dfrac{e^{\eta_ix^{(i)}}}{\sum_{j = 1}^{k}e^{\eta_jx^{(i)}} })^{1\{y^{(i)} = w\}}$<p>
</p>
</font>
</div>