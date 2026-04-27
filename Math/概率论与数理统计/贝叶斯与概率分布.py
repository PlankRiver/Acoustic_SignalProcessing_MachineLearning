#贝叶斯
#P(Bi|A)=P(A|Bi)/sum(P(A|Bj))
#P(A|B)=P(AB)/P(B)
#P(A)=sum(P(A|Bi)*p(Bi))
#P(A|B)*P(B)=P(B|A)*P(A)

#两点分布
#x: 1 0
#P: p 1-p
#E(x) = p
#D(x) = p(1-p)

#二项分布 抛n次硬币
#E(x) = np
#D(x) = np(1-p)

#负二项分布
#一系列的独立成败实验反复试验，直到成功r次停止，试验次数为x的概率P(X=x,r,p)=C(r-1)(x-1)p^r*(1-p)^(x-r)

#泊松分布 (数数相关)
#P(X=k)=x^k*e^-x/k!
#E(X)=x

#均匀分布
#f(x)=1/(b-a),x<[a,b]

#指数分布  自由程分布 无记忆性
#f(x)=e^(-x/x0)/x0

#正态分布
#f(x)=e^(-(x-mv)^2/(2*sigma^2))/(sqrt(2*pi)*sigma)

#期望
#无条件成立  E(kX)=kE(X)  E(X+Y)=E(X)+E(Y)
#独立事件  E(XY)=E(X)E(Y)  反之不然
#方差
#Var(X)=E(X^2)-E(X)^2
#无条件成立  Var(c)=0  Var(X+c)=Var(X) Var(kX)=k^2Var(X)
#独立事件  Var(X+Y)=Var(X)+Var(Y)  Var(XY)=Var(X)*Var(Y)+Var(X)*E(Y)^2+Var(Y)*E(X)^2

#协方差
#Cov(X,Y)=E([X-E(X)][Y-E(Y)])
#Cov(X,Y)=Cov(Y,X)
#Coc(aX+b,cY+d)=acCov(E,Y)
#Cov(X1+X2,Y)=Cov(X1,Y)+Cov(X2,Y)
#Cov(X,Y)=E(XY)-E(X)E(Y)
#协方差的意:度量两个量变化趋势相同的量
#Cov>0 变化趋势相同
#Cov<0 变化趋势相反
#Cov=0 不相关
#|Cov(X,Y)|=sigma(X)*sigma(Y) 当且仅当XY呈线性关系时等号成立   证明过程类似柯西不等式，构造二次函数使用判别式大于0
#协方差矩阵
#对于n个随机向量(X1,X2,X3,...,Xn),有C=(Cov(Xi,Xj))n*n  (对称矩阵)

#Pearson相关系数
#p=Cov(X,Y)/sqrt(Var(X)*Var(Y))  p<=1

#切比雪夫不等式
#P{|X-mv|>epsilon}<=Var(X)/epsilon^2

#大数定理
#一系列随机变量X1,X2,X3,...,Xn,...具有相同的均值mv和相同的方差sigma^2，取前n个变量的均值Yn=sum(Xi)(1<=i<=n)/n
#则有lim(n->oo)P{|Yn-mv|<epsilon}=1
#推论lim(n->oo)P{|nA/n-p|<epsilon}=1  事件A的概率为p

#中心极限定理
#一系列随机变量X1,X2,X3,...,Xn,...具有相同的均值mv和相同的方差sigma^2
#取Yn=(sum(xi)(1<=i<=n)-n*mv)/(sqrt(n)*sigma)将服从标准正态分布N(0,1),sum(xi)(1<=i<=n)服从正态分布N(n*mv,n*sigma^2)
#现实中，许多独立现象可以看成许多因素独立影响的综合反映，往往近似服从正态分布

#最大似然估计  可能会发生过拟合现象
#某联合事件发生的概率L(xi;q1,q2,...,1k)=multiply(f(xi;qi))(1<=i<=n)
#最大似然:ln(L)取极值,将猜测参数求出