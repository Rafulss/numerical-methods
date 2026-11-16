import numpy as np
import matplotlib.pyplot as plt

g = 9.81
l = 100

theta = 0.5
N = 20
M = 201

max_iter = 10000

n = np.arange(-N, N + 1)
omega0 = np.sqrt(g / l)

T = 1.3 * 2 * np.pi
omega = 2 * np.pi / T

t = np.linspace(0, T, M, endpoint=False)

An = np.where(np.abs(n) == 1, theta / 2, 0).astype(complex)

Iter = []
amplituda = []

def alfan_k():
    C = np.zeros(M, dtype=complex)
    C[n % M] = An
    return M * np.fft.ifft(C)

error = 1e-10
error0 = np.inf
k = 0

while error0 >= error and k < max_iter:
    alfa_k = alfan_k()
    amplituda.append(np.max(np.abs(alfa_k.real)))
    Iter.append(k)

    B_all = np.fft.fft(np.sin(alfa_k)) / M
    B = B_all[n % M]

    A_new = np.zeros_like(An)
    nonzero = n != 0
    A_new[nonzero] = B[nonzero] / (omega**2 * n[nonzero]**2)

    error0 = np.max(np.abs(A_new - An))
    An = A_new
    k += 1

alfa_k = alfan_k()
gala_amplituda = np.max(np.abs(alfa_k.real))

Iter.append(k)
amplituda.append(gala_amplituda)

x = np.array(Iter)
y = np.array(amplituda)

amplitudas_kluda = np.abs(y[-1] - y)

plt.figure(figsize=(6,4))
plt.plot(x, amplitudas_kluda, "o-", markersize=4)
plt.yscale("log")
plt.xlabel("Iterācijas numurs, lin")
plt.ylabel(r"Kļūda (rad), log")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(6,4))
plt.plot(x, amplitudas_kluda, "o-", markersize=4)
plt.xlabel("Iterācijas numurs, lin")
plt.ylabel(r"Kļūda (rad), lin")
plt.grid()
plt.tight_layout()

plt.show()