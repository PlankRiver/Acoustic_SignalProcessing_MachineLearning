import numpy
import matplotlib.pyplot as plt

# 使用四阶 Runge-Kutta 方法求解类 Goodwin 昼夜节律振荡器模型。
# 状态变量：
# - M：mRNA 浓度
# - P0、P1、P2：不同磷酸化状态下的蛋白质浓度
# - Pn：细胞核内蛋白质浓度

# 模型参数
vs = 0.76
vm = 0.65
vd = 0.95
ks = 0.38
k1 = 1.9
k2 = 1.3
v1 = 3.2
v2 = 1.58
v3 = 5.0
v4 = 2.5
K1 = 2.0
K2 = 2.0
K3 = 2.0
K4 = 2.0
Ki = 1.0
Km = 0.5
Kd = 0.2

# 初始化
# dt：积分时间步长
# n：积分步数
t = 0.0
dt = 0.1
n = 10000
M = 1.9
P0 = 0.8
P1 = 0.8
P2 = 0.8
Pn = 0.8
P_t = P0+P1+P2+Pn

# dM/dt
def fM(M,Pn):
    freturn = vs*Ki**4/(Ki**4+Pn**4)-vm*M/(Km+M)
    return freturn

# dP0/dt
def fP0(M,P0,P1):
    freturn = ks*M-v1*P0/(K1+P0)+v2*P1/(K2+P1)
    return freturn

# dP1/dt
def fP1(P0,P1,P2):
    freturn = v1*P0/(K1+P0)-v2*P1/(K2+P1)-v3*P1/(K3+P1)+v4*P2/(K4+P2)
    return freturn

# dP2/dt
def fP2(P1,P2,Pn):
    freturn = v3*P1/(K3+P1)-v4*P2/(K4+P2)-k1*P2+k2*Pn-vd*P2/(Kd+P2)
    return freturn

# dPn/dt
def fPn(P2,Pn):
    freturn = k1*P2-k2*Pn
    return freturn

M_RK4 = []; M_RK4.append(M)
P0_RK4 = []; P0_RK4.append(P0)
P1_RK4 = []; P1_RK4.append(P1)
P2_RK4 = []; P2_RK4.append(P2)
Pn_RK4 = []; Pn_RK4.append(Pn)
P_t_RK4 = []; P_t_RK4.append(P_t)
time = []; time.append(t)

# 对耦合常微分方程组执行一次四阶 Runge-Kutta 积分。
def RK4(M,P0,P1,P2,Pn):
    # 第 1 阶段斜率
    kM_1 = fM(M,Pn)
    kP0_1 = fP0(M,P0,P1)
    kP1_1 = fP1(P0,P1,P2)
    kP2_1 = fP2(P1,P2,Pn)
    kPn_1 = fPn(P2,Pn)

    # 第 2 阶段斜率（半步预测）
    kM_2 = fM(M+kM_1*dt/2,Pn+kPn_1*dt/2)
    kP0_2 = fP0(M+kM_1*dt/2,P0+kP0_1*dt/2,P1+kP1_1*dt/2)
    kP1_2 = fP1(P0+kP0_1*dt/2,P1+kP1_1*dt/2,P2+kP2_1*dt/2)
    kP2_2 = fP2(P1+kP1_1*dt/2,P2+kP2_1*dt/2,Pn+kPn_1*dt/2)
    kPn_2 = fPn(P2+kP2_1*dt/2,Pn+kPn_1*dt/2)

    # 第 3 阶段斜率（半步预测）
    kM_3 = fM(M+kM_2*dt/2,Pn+kPn_2*dt/2)
    kP0_3 = fP0(M+kM_2*dt/2,P0+kP0_2*dt/2,P1+kP1_2*dt/2)
    kP1_3 = fP1(P0+kP0_2*dt/2,P1+kP1_2*dt/2,P2+kP2_2*dt/2)
    kP2_3 = fP2(P1+kP1_2*dt/2,P2+kP2_2*dt/2,Pn+kPn_2*dt/2)
    kPn_3 = fPn(P2+kP2_2*dt/2,Pn+kPn_2*dt/2)

    # 第 4 阶段斜率（整步预测）
    kM_4 = fM(M+kM_3*dt,Pn+kPn_3*dt)
    kP0_4 = fP0(M+kM_3*dt,P0+kP0_3*dt,P1+kP1_3*dt)
    kP1_4 = fP1(P0+kP0_3*dt,P1+kP1_3*dt,P2+kP2_3*dt)
    kP2_4 = fP2(P1+kP1_3*dt,P2+kP2_3*dt,Pn+kPn_3*dt)
    kPn_4 = fPn(P2+kP2_3*dt,Pn+kPn_3*dt)

    # 加权平均斜率
    kM = (kM_1+2*kM_2+2*kM_3+kM_4)/6
    kP0 = (kP0_1+2*kP0_2+2*kP0_3+kP0_4)/6
    kP1 = (kP1_1+2*kP1_2+2*kP1_3+kP1_4)/6
    kP2 = (kP2_1+2*kP2_2+2*kP2_3+kP2_4)/6
    kPn = (kPn_1+2*kPn_2+2*kPn_3+kPn_4)/6

    # 更新状态变量
    M += kM*dt
    P0 += kP0*dt
    P1 += kP1*dt
    P2 += kP2*dt
    Pn += kPn*dt

    return(M,P0,P1,P2,Pn)

for i in range(n):
    M,P0,P1,P2,Pn = RK4(M,P0,P1,P2,Pn)
    t += dt
    P_t = P0+P1+P2+Pn
    M_RK4.append(M)
    P0_RK4.append(P0)
    P1_RK4.append(P1)
    P2_RK4.append(P2)
    Pn_RK4.append(Pn)
    P_t_RK4.append(P_t)
    time.append(t)

# 绘制时间序列图和相图。
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(time,M_RK4,'r-',label='M',linewidth=1.0)
ax1.plot(time,P0_RK4,'b-',label='M',linewidth=1.0)
ax1.plot(time,P1_RK4,'c-',label='M',linewidth=1.0)
ax1.plot(time,P2_RK4,'g-',label='M',linewidth=1.0)
ax1.plot(time,Pn_RK4,'y-',label='M',linewidth=1.0)
ax1.plot(time,P_t_RK4,'k-',label='M',linewidth=1.0)

ax2.plot(P_t_RK4,M_RK4,'r-',label='Phase',linewidth=1.0)

ax1.set_ylabel('PER forms of M',fontsize=20)
ax1.set_xlabel(r'Time/h',fontsize=20)
ax1.set_xlim(200,300)
ax1.set_ylim(0,5)

ax2.set_ylabel(r'mRNA',fontsize=20)
ax2.set_xlabel(r'Total PER',fontsize=20)
ax2.set_xlim(0,6)
ax2.set_ylim(0,4)

# 保存高分辨率图片，并以交互方式显示。
plt.savefig('The simulation of biotic period.png', dpi=300, bbox_inches='tight')
plt.show()
