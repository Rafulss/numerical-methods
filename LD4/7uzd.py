import numpy as np
import matplotlib.pyplot as plt

N = 1000
xi_max = 5

# Fizikālās konstantes SI vienībās
hbar = 1.054571817e-34  # J·s
h_planck = 2 * np.pi * hbar
c_light = 299792458    # m/s
u = 1.66053906660e-27  # kg

# Morzes potenciāla parametri SI vienībās
D_e = h_planck * c_light * (79890 * 100)  # cm⁻¹ -> m⁻¹ -> J
k = 2.6889 * 1e10                        # Å⁻¹ -> m⁻¹

# Slāpekļa atoma masa (aptuveni 14 u) un reducētā masa
m = 14.0 * u
mu = m / 2

# Frekvence un mērogi
omega = k * np.sqrt(2 * D_e / mu)
x0 = np.sqrt(hbar / (mu * omega))  # garuma mērogs, m
E0 = hbar * omega                 # enerģijas mērogs, J


i = np.linspace(-N, N, 2*N+1) #norādu cik daudz vērtības gribu
xi_i = i * xi_max / N  #definēju koordinātas

# definējam xi bez gala punktiem, jo tur psi=0
xi_inner = xi_i[1:-1] #visi iņemot 1. un pēdējo xi_i
M = len(xi_inner)  # Nezināmo skaits: 2N - 1

# izmantojam galīgo diferenču formulu, lai noteiktu ieguldījuma koeficientus
kinetic_center = 1.0 / (xi_max/N)**2  #reizina psi_i
kinetic_neighbor = -1.0 / (2.0 * (xi_max/N)**2)  #reizina psi_i-1 un psi_i+1

# pēc diskretizācijas iegūta ipvert problēma Hpsi=Epsi
H = np.zeros((M, M))  #H matrica M*M

# H diag ir kin un pot ieguldījums, tieši blakus psi_i+-1
for j in range(M):
    # Potenciāls attiecīgajā iekšējā režģa punktā
    # x = x0 * xi; potenciālu dalām ar E0
    potential = (D_e / E0) * (
        1 - np.exp(-k * x0 * xi_inner[j])
    )**2

    # Galvenā diagonāle
    H[j, j] = kinetic_center + potential

    # Elementi tieši virs un zem galvenās diagonāles
    if j < M - 1:
        H[j, j + 1] = kinetic_neighbor
        H[j + 1, j] = kinetic_neighbor

# iegūstam īpenerg un īpvekt
eigenvalues, eigenvectors = np.linalg.eigh(H)

for n in range(5):
    print(f"n = {n}, epsilon = {eigenvalues[n]:.8f}")

# 3.uzd 5 mazākās eigenvalues
print("5 mazākās īpašvērtības:", eigenvalues[:5])