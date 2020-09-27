[《Fast Algorithms for Convolutional Neural Networks》](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)

## 1. 从 $F(n, r)$ 开始
   F表示FIR（有限脉冲响应）。n表示输出的长度，r表示参数的长度，那么输入的长度是多少呢？这里默认$stride=1$，输入的长度$alpha = n + r - 1$。
   
   正常的情况下，F接收$X[n]$的输入信号，依次输出$Y[n]$。那么$F(n, r)$的乘法数量$n*r$。

   winograd变换将卷积的运算形式变成输入和参数的element-wise乘法。不考虑变换过程中的运算，winograd变换后$F(n, r)$的乘法数量就是输入的长度$n+r-1$

   那么变换后的乘法数量的加速比就是$\frac{n*r}{n+r-1}$

   ![](/home/zhuxl/Desktop/Selection_051.png)  ![](/home/zhuxl/Desktop/Selection_052.png)

   可以看到参数长度为3时，加速比的极值为3，参数长度为7时，加速比的极值时7。从这里看，大的卷积核，加速比的上限越高。但是为什么会说winograd算法只适合小的卷积核呢，这里还需要考虑缓存的上限。如果数据不能放在缓存，而是放在内存了，那么估计数据读取的时间也够完成很多次乘法了。大的卷积核对应需要缓存的数据量$(n+r-1)$就更多一些。而且在总数一定的情况下（缓存限制），r越大n就越小，n越小，加速比就越小。

   上面都说的是一维的情况，二维卷积时也是类似的。$F(m*n, r*s)$,需要的输入tile大小为$(m+r-1)*(n+s-1)$，乘法数量的加速比就是$\frac{m*n*r*s}{(m+r-1)*(n+s-1)}$。

## 2. $Y=A^{T}[[GgG^{T}]\odot[B^{T}dB]]A$
  这里的难点在于根据求$F(n,r)$对应的$A^{T},G,B^{T}$。文章中作者说根据中国余数理论来求的，请让我哭一会儿，完全不会。不过，在[《卷积神经网络中的winograd》](https://www.cnblogs.com/shine-lee/p/10906535.html#%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84winograd)文章中找到了[wincnn](https://github.com/andravin/wincnn.git)。使用这个工具可以方便的计算任意$F(n,r)$对应的变换矩阵$A^{T},G,B^{T}$。

  一维卷积使用公式$Y=A^{T}[(Gg)\odot(B^{T}d)]$。$A^{T}$的大小为$n*alpha$，G的大小为$alpha*r$，$B^{T}$的大小为$alpha * alpha$，$alpha=n+r-1$。
  
  二维卷积时，输出和卷积核都是方的情况下$F(n*n, r*r)$，$F(n,r)$求得$A^{T},G,B^{T}$。$Y=A^{T}[[GgG^{T}]\odot[B^{T}dB]]A$。
  
  如果输出和卷积核不是方的，$F(m*n, r*s)$，我猜测公式可以扩展成$Y=A^{T}_{0}[[G_{0}gG^{T}_{1}]\odot[B^{T}_{0}dB_{1}]]A_{1}$；$F(m,r)$对应$A^{T}_{0},G_{0},B^{T}_{0}$，$F(n,s)$对应$A^{T}_{1},G_{1},B^{T}_{1}$。

  另外Y的计算是在h,w的维度上进行的，卷积核还有out_channels和in_channel。并且需要在in_channel的维度需要对Y求和。假设$Y_{i}=\Sigma^{in_channels}_{j}(A^{T}M_jA)$，那么也可以先求$M=\Sigma^{in_channels}_{j}M_j$，再计算$A^{T}MA$，即$Y_{i}=A^{T}(\Sigma^{in_channels}_{j}M_j)A$，这样可以减少output_transform的次数。

## CPU实现


## to-do list
  1. stride！= 1的情况
  2. dilation！=1的情况
  3. 3D卷积