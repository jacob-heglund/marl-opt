import numpy as np

def f_1(x):
    return np.abs(x**3 - x**2)


lambda_vals = np.linspace(0, 1, num=10)

print(f_1(lambda_vals))