import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal, eig_banded


def calculate_eigenpairs(N, xi_max, points=3, vectors=False):
    """Piecas zemākās enerģijas un, ja prasīts, īpašvektori."""
    h = xi_max / N
    xi_inner = np.arange(-N + 1, N) * h
    M = len(xi_inner)

    if M < 5:
        raise ValueError("Nepieciešami vismaz pieci iekšējie punkti.")

    potential = xi_inner**2 / 2.0

    if points == 3:
        diagonal = 1.0 / h**2 + potential
        off_diagonal = np.full(M - 1, -1.0 / (2.0 * h**2))

        return eigh_tridiagonal(
            diagonal,
            off_diagonal,
            eigvals_only=not vectors,
            select="i",
            select_range=(0, 4)
        )

    if points == 5:
        diagonal = 5.0 / (4.0 * h**2) + potential

        # Nepāra turpinājums pāri nulles robežnosacījumiem.
        diagonal[0] -= 1.0 / (24.0 * h**2)
        diagonal[-1] -= 1.0 / (24.0 * h**2)

        first_off = -2.0 / (3.0 * h**2)
        second_off = 1.0 / (24.0 * h**2)

        # Simetriskas joslu matricas augšējās daļas glabāšana:
        # ab[2, :]   — galvenā diagonāle;
        # ab[1, 1:]  — pirmā augšējā diagonāle;
        # ab[0, 2:]  — otrā augšējā diagonāle.
        ab = np.zeros((3, M))
        ab[2, :] = diagonal
        ab[1, 1:] = first_off
        ab[0, 2:] = second_off

        return eig_banded(
            ab,
            lower=False,
            eigvals_only=not vectors,
            select="i",
            select_range=(0, 4)
        )

    raise ValueError("points jābūt 3 vai 5.")


exact_energies = np.arange(5) + 0.5
N_check = 1000
xi_max_check = 5.0

# Piecu punktu metode: īpašvērtības un īpašvektori.
energies_5, eigenvectors_5 = calculate_eigenpairs(
    N_check, xi_max_check, points=5, vectors=True
)

# Normētās bezdimensiju viļņu funkcijas iekšējos punktos.
h = xi_max_check / N_check
psi_5 = eigenvectors_5 / np.sqrt(h)

energies_3 = calculate_eigenpairs(
    N_check, xi_max_check, points=3
)

print(f"\nN = {N_check}, xi_max = {xi_max_check:g}")
print("n   Teorija     3 punktu enerģija   5 punktu enerģija"
      "   Kļūda (3 p.)   Kļūda (5 p.)")

for n in range(5):
    print(
        f"{n}   {exact_energies[n]:.1f}         "
        f"{energies_3[n]:.10f}       "
        f"{energies_5[n]:.10f}       "
        f"{abs(energies_3[n] - exact_energies[n]):.3e}      "
        f"{abs(energies_5[n] - exact_energies[n]):.3e}"
    )


# 1. Kļūdu atkarība no N pie xi_max = 5.
N_values = np.arange(10, 1001, 10)

# 2. Kļūdu atkarība no xi_max pie N = 1000.
xi_max_values = np.linspace(1.0, 10.0, 91)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

graph_settings = [
    (
        N_values,
        lambda N, points: calculate_eigenpairs(
            int(N), xi_max_check, points=points
        ),
        r"$N$",
        r"Kļūdas atkarībā no $N$, $\xi_{\max}=5$",
        "5uzd_N_3un5.png"
    ),
    (
        xi_max_values,
        lambda xi_max, points: calculate_eigenpairs(
            N_check, float(xi_max), points=points
        ),
        r"$\xi_{\max}$",
        r"Kļūdas atkarībā no $\xi_{\max}$, $N=1000$",
        "5uzd_ximax_3un5.png"
    )
]

for values, calculator, xlabel, title, filename in graph_settings:
    fig, ax = plt.subplots(figsize=(10, 6))

    for points, linestyle in [(3, "-"), (5, "--")]:
        energies = np.array([
            calculator(value, points) for value in values
        ])
        errors = np.abs(energies - exact_energies)

        # Precīzi nulles kļūdu logaritmiskā skalā nevar attēlot.
        errors[errors == 0.0] = np.nan

        for n in range(5):
            ax.semilogy(
                values,
                errors[:, n],
                color=colors[n],
                linestyle=linestyle,
                linewidth=1.4,
                label=f"n = {n}, {points} punkti"
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$|\varepsilon_n-(n+1/2)|$")
    ax.grid(True, which="major", alpha=0.6)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()


# 3. Abu metožu konverģences pārbaude.
for points in (3, 5):
    base = calculate_eigenpairs(1000, 5.0, points=points)
    finer = calculate_eigenpairs(2000, 5.0, points=points)
    wider = calculate_eigenpairs(1200, 6.0, points=points)

    print(f"\nKonverģences pārbaude: {points} punktu formula")
    print("n   Absolūtā kļūda   Izmaiņa, N=2000"
          "   Izmaiņa, xi_max=6")

    for n in range(5):
        print(
            f"{n}   {abs(base[n] - exact_energies[n]):.3e}        "
            f"{abs(finer[n] - base[n]):.3e}          "
            f"{abs(wider[n] - base[n]):.3e}"
        )

plt.show()