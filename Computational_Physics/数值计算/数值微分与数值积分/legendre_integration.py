import numpy as np
from scipy.special import roots_legendre
import time

EPS_REG = 1e-8
TOL = 1e-8


def legendre_integration(a, b, f, n):
    x, w = roots_legendre(n)
    x = 0.5 * (b - a) * x + 0.5 * (b + a)
    w = 0.5 * (b - a) * w
    return np.sum(w * f(x))


def f(x, eps_reg=EPS_REG):
    return np.exp(-x**2) / (np.sqrt((x - 1.5)**2) + eps_reg)


def integrate_with_n(f, a, b, n):
    return legendre_integration(a, b, f, n)


def precise_integration(f, a, b, tol=TOL, n0=2, max_n=1 << 16):
    n = n0
    while n <= max_n:
        integration_1 = integrate_with_n(f, a, b, n)
        integration_2 = integrate_with_n(f, a, b, 2 * n)
        error = abs(integration_1 - integration_2)
        n *= 2
        if error < tol:
            return integration_1, n//2, error
    raise RuntimeError(f"Failed to achieve desired precision with max_n={max_n}")

if __name__ == "__main__":
    start_time = time.time()

    a, b = -1.0, 1.0
    result, n, error = precise_integration(f, a, b)
    print(f"Integral result: {result}, Number of points: {n}, Estimated error: {error}")

    print(f'Total time: {time.time() - start_time:.10f} seconds')