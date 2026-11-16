import numpy as np
import matplotlib.pyplot as plt

x_max = 5  # Bezdimensionālā aprēķinu intervāla pusgarums
N = 10000  # Iekšējo režģa punktu skaits
VertibuSkaits = 5

h = 2 * x_max / (N + 1)
xi = np.linspace(-x_max, x_max, N + 2)[1:-1]

H = np.zeros((N, N))

for i in range(N):
    H[i, i] = 1 / h**2 + xi[i]**2 / 2

for i in range(N - 1):
    H[i, i + 1] = -1 / (2 * h**2)
    H[i + 1, i] = -1 / (2 * h**2)

E, psi = np.linalg.eigh(H)

n = np.arange(5)
E_sk = E[:5]
E_teor = n + 0.5

kluda = np.abs(E_sk - E_teor)
rel_kluda = kluda / E_teor * 100

print("\nPiecas mazākās bezdimensionālās īpašvērtības")
print(f"x_max = {x_max}, N = {N}, h = {h:.6f}\n")

print(
    f"{'n':>3} {'Skaitliskā':>14} {'Teorētiskā':>14} "
    f"{'Absolūtā kļūda':>17} {'Rel. kļūda (%)':>17}"
)
print("-" * 69)

for i in range(5):
    print(
        f"{n[i]:3d} {E_sk[i]:14.8f} {E_teor[i]:14.8f} "
        f"{kluda[i]:17.8f} {rel_kluda[i]:17.6f}"
    )

dati = []

for i in range(VertibuSkaits):
    dati.append([
        str(n[i]),
        f"{E_sk[i]:.8f}",
        f"{E_teor[i]:.8f}",
        f"{kluda[i]:.8f}",
        f"{rel_kluda[i]:.6f}"
    ])
