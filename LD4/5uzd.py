import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


def calculate_eigenvalues(N, xi_max):
    h = xi_max / N

    # Iekšējie punkti; galapunktos psi = 0
    xi_inner = np.arange(-N + 1, N) * h
    M = len(xi_inner)  # Nezināmo skaits: 2N - 1

    # Tā pati Hamiltona matrica kā 4. uzdevumā,
    # tikai glabājam galveno un blakus diagonāli.
    diagonal = 1.0 / h**2 + xi_inner**2 / 2.0
    off_diagonal = np.full(M - 1, -1.0 / (2.0 * h**2))

    # Aprēķinām tikai piecas mazākās īpašvērtības.
    return eigh_tridiagonal(
        diagonal,
        off_diagonal,
        eigvals_only=True,
        select="i",
        select_range=(0, 4)
    )


# Precīzās pirmās piecas bezdimensiju īpašenerģijas
exact_energies = np.arange(5) + 0.5

# 3. un 4. uzdevumā izmantotie parametri
N_check = 1000
xi_max_check = 5.0

energies_check = calculate_eigenvalues(N_check, xi_max_check)
errors_check = np.abs(energies_check - exact_energies)


# 1. Mainām N, fiksējot xi_max = 5
N_i = np.arange(10, 1001, 10)  # N = 10, 20, ..., 1000
xi_max_fix = xi_max_check

energies_N = np.array([
    calculate_eigenvalues(int(N), xi_max_fix)
    for N in N_i
])

errors_N = np.abs(energies_N - exact_energies)

fig, ax = plt.subplots(figsize=(8, 5))

for n in range(5):
    ax.semilogy(
        N_i,
        errors_N[:, n],
        "o-",
        markersize=3,
        label=f"n = {n}"
    )

ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$|\varepsilon_n-(n+1/2)|$")
ax.set_title(
    rf"Enerģiju kļūdas atkarībā no $N$, "
    rf"$\xi_{{\max}}={xi_max_fix:g}$"
)

ax.set_xlim(0, 1050)
ax.set_xticks(np.arange(0, 1001, 100))
ax.legend()
ax.grid()

fig.tight_layout()


# 2. Mainām xi_max, fiksējot N = 1000
xi_max_i = np.linspace(1.0, 10.0, 91)
N_fix = N_check

energies_xi = np.array([
    calculate_eigenvalues(N_fix, xi_max)
    for xi_max in xi_max_i
])

errors_xi = np.abs(energies_xi - exact_energies)

fig, ax = plt.subplots(figsize=(8, 5))

for n in range(5):
    ax.semilogy(
        xi_max_i,
        errors_xi[:, n],
        "o-",
        markersize=3,
        label=f"n = {n}"
    )

ax.set_xlabel(r"$\xi_{\max}$")
ax.set_ylabel(r"$|\varepsilon_n-(n+1/2)|$")
ax.set_title(
    rf"Enerģiju kļūdas atkarībā no $\xi_{{\max}}$, "
    rf"$N={N_fix}$"
)

ax.legend()
ax.grid()

fig.tight_layout()


# 3. Papildu konverģences pārbaude

# Samazinām režģa soli divas reizes, saglabājot intervālu.
energies_finer = calculate_eigenvalues(2000, 5.0)

# Paplašinām intervālu, saglabājot režģa soli h = 0.005.
energies_wider = calculate_eigenvalues(1200, 6.0)

change_N = np.abs(energies_finer - energies_check)
change_interval = np.abs(energies_wider - energies_check)


# 4. Izvadām aprēķinātās enerģijas un kļūdas
print(f"\nN = {N_check}, xi_max = {xi_max_check:g}")
print("n    Aprēķinātā enerģija    Precīzā enerģija    Absolūtā kļūda")

for n in range(5):
    print(
        f"{n}    {energies_check[n]:.10f}            "
        f"{exact_energies[n]:.1f}                "
        f"{errors_check[n]:.3e}"
    )

print("\nKonverģences pārbaude")
print("n    Absolūtā kļūda    Izmaiņa, N=2000    Izmaiņa, xi_max=6")

for n in range(5):
    print(
        f"{n}    {errors_check[n]:.3e}          "
        f"{change_N[n]:.3e}            "
        f"{change_interval[n]:.3e}"
    )


# Parādām abus grafikus.
plt.show()