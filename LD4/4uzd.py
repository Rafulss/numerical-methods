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

# iegūstam īp-energ un īp-vekt
eigenvalues, eigenvectors = np.linalg.eigh(H)

eigenvalues5 = eigenvalues[:5]  #pirmās 5 eigenvalues
n = np.arange(5)

# rēķinām īp-f-jas
psi = np.zeros((2 * N + 1, M)) # Katra rinda - koordināta, kolonna - H matricas īp-vektori bez galap-tiem

# eigh īpašvektoriem summa |v_i|^2 = 1, tāpēc integrāļa normalizācijai dalām ar sqrt(h)
psi[1:-1, :] = eigenvectors / np.sqrt(xi_max / N)

psi_0 = psi[:, 0]  #pamatstāvokļa viļņu funkcija
psi5 = psi[:, :5]  #pirmās 5 īp-f-jas atrodas kolonnas

A = 0.5  # viļņu funkciju attēlojuma amplitūdas mērogs

for j in n:
    plt.plot(
        xi_i,
        eigenvalues5[j] + A * psi5[:, j],
        label=f"n = {j}, ε = {eigenvalues5[j]:.3f}"
    )
    plt.axhline(
        eigenvalues5[j],
        color="black",
        linestyle="--",
        linewidth=0.7
    )

plt.xlabel("Bezdimensiju koordināta ξ")
plt.ylabel("εₙ + Aψₙ(ξ)")
plt.legend()
plt.grid()
plt.show()