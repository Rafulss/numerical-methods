import numpy as np
import matplotlib.pyplot as plt

a0 = np.arange(1, 11) * np.pi / 20
N = 100
T = 10.0

h = T / N
t = np.linspace(0, T, N + 1)

alpha = a0[0]

n_int = N - 1
x = np.zeros((n_int, n_int))

for i in range(n_int):
    x[i, i] = -2 / h**2 + omega**2
    if i > 0:
        x[i, i - 1] = 1 / h**2
    if i < n_int - 1:
        x[i, i + 1] = 1 / h**2

print(x)


