import numpy as np
import matplotlib.pyplot as plt

N = 1000
xi_max = 5

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
    potential = xi_inner[j]**2 / 2.0

    # Galvenā diagonāle
    H[j, j] = kinetic_center + potential

    # Elementi tieši virs un zem galvenās diagonāles
    if j < M - 1:
        H[j, j + 1] = kinetic_neighbor
        H[j + 1, j] = kinetic_neighbor

# iegūstam īpenerg un īpvekt
eigenvalues, eigenvectors = np.linalg.eigh(H)

for n in range(6):
    print(f"n = {n}, epsilon = {eigenvalues[n]:.8f}")

# 3.uzd 5 mazākās eigenvalues
print("5 mazākās īpašvērtības:", eigenvalues[:5])