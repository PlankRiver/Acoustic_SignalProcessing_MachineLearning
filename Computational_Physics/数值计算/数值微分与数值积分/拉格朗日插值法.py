import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x, Y, X):
    n = len(X)
    y = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p *= (x-X[j])/(X[i]-X[j])
        y += p*Y[i]
    return y

def lagrange_piecewise_interpolation(x_eval, Y, X, points_per_seg=3):
    stride = points_per_seg - 1
    n_points = len(X)
    y_result = np.zeros_like(x_eval, dtype=float)
    
    for i in range(0, n_points - 1, stride):
        X_seg = X[i : i + points_per_seg]
        Y_seg = Y[i : i + points_per_seg]
        
        x_min = X_seg[0]
        x_max = X_seg[-1]
        
        if i + points_per_seg >= n_points:
            mask = (x_eval >= x_min) & (x_eval <= x_max)
        else:
            mask = (x_eval >= x_min) & (x_eval < x_max)
        
        if not np.any(mask):
            continue
            
        y_result[mask] = lagrange_interpolation(x_eval[mask], Y_seg, X_seg)
        
    return y_result

def cubic_spline_interpolation(x_eval, Y, X):
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(X, Y)
    return cs(x_eval)

if __name__ == '__main__':
    # 原始数据
    S = np.array([1390, 1425, 1461, 1620, 1665])
    E = np.array([5156, 5440, 5486, 5736, 5805])
    
    # 生成密集的测试点用于平滑绘图
    x = np.linspace(1390, 1665, 500)
    
    # 计算三种插值结果
    y_global = lagrange_interpolation(x, E, S)
    y_piecewise = lagrange_piecewise_interpolation(x, E, S, points_per_seg=3)
    y_spline = cubic_spline_interpolation(x, E, S)
    
    # 开始绘图
    plt.figure(figsize=(10, 6)) # 稍微调大一点画布，看起来更清晰
    
    # 1. 全局拉格朗日插值 (Global Lagrange)
    plt.plot(x, y_global, color='green', linestyle=':', linewidth=2, label='Global Lagrange (Degree 4)')
    
    # 2. 分段拉格朗日插值 (Piecewise Quadratic)
    # 因为5个点，step=3，刚好分成两段二次曲线：[0,1,2] 和 [2,3,4]
    plt.plot(x, y_piecewise, color='orange', linestyle='--', linewidth=2, label='Piecewise Lagrange (Quadratic)')
    
    # 3. 三次样条插值 (Cubic Spline)
    plt.plot(x, y_spline, color='blue', linestyle='-', linewidth=2, alpha=0.7, label='Cubic Spline')
    
    # 绘制原始数据点
    plt.plot(S, E, 'ro', markersize=8, zorder=5, label='Data Points')
    
    # 设置图表属性
    plt.xlabel('S (Variable)')
    plt.ylabel('E (Value)')
    plt.title('Comparison of Interpolation Methods')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 显示图像
    plt.tight_layout()
    plt.show()