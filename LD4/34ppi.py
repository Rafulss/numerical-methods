import numpy as np
from scipy.linalg import eigh_tridiagonal

# Tās pašas konstantes un parametri kā 7. uzdevumā.
hbar = 1.054571817e-34
c_light = 299792458
u = 1.66053906660e-27

D_e = 2 * np.pi * hbar * c_light * 79890 * 100
k = 2.6889e10
mu = 7 * u

omega = k * np.sqrt(2 * D_e / mu)
x0 = np.sqrt(hbar / (mu * omega))
E0 = hbar * omega

threshold = D_e / E0
beta = k * x0
lam = np.sqrt(2 * mu * D_e) / (hbar * k)

# Atļauti tikai stāvokļi, kuriem n + 1/2 < lambda.
n_bound = np.arange(int(np.ceil(lam)))
n_bound = n_bound[n_bound + 0.5 < lam]

analytic_bound = (
    n_bound + 0.5
    - (n_bound + 0.5)**2 / (2 * lam)
)

print(f"lambda = {lam:.8f}")
print(f"Disociācijas slieksnis = {threshold:.8f}")
print(f"Analītiskais saistīto stāvokļu skaits = {len(n_bound)}")


def morse_diagonals(xi_min, xi_max, step):
    # Vienmērīgs režģis ar precīzi iekļautiem galapunktiem.
    intervals = int(round((xi_max - xi_min) / step))
    xi = np.linspace(xi_min, xi_max, intervals + 1)
    h = xi[1] - xi[0]
    xi_inner = xi[1:-1]

    potential = threshold * (
        1 - np.exp(-beta * xi_inner)
    )**2

    diagonal = 1 / h**2 + potential
    off_diagonal = np.full(
        len(xi_inner) - 1,
        -1 / (2 * h**2)
    )

    return diagonal, off_diagonal


# 1. Saistīto stāvokļu skaita skaitliskā pārbaude.
# Paplašinām intervālu un pēc tam samazinām režģa soli.
settings = [
    (-10.0, 200.0, 0.02),
    (-10.0, 400.0, 0.02),
    (-10.0, 400.0, 0.01),
    (-10.0, 400.0, 0.005)
]

print("\nSaistīto stāvokļu pārbaude")
print("xi_max    solis    skaits    augstākā saistītā enerģija")

for xi_min, xi_max, step in settings:
    diagonal, off_diagonal = morse_diagonals(
        xi_min, xi_max, step
    )

    # Aprēķinām tikai īpašvērtības zem disociācijas sliekšņa.
    upper = np.nextafter(threshold, -np.inf)

    bound_energies = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        eigvals_only=True,
        select="v",
        select_range=(0.0, upper)
    )

    print(
        f"{xi_max:6.0f}    {step:.3f}    "
        f"{len(bound_energies):6d}    "
        f"{bound_energies[-1]:.10f}"
    )


# 2. Salīdzinājums tieši ar 7. uzdevuma aprēķinu:
# intervāls [-5, 5], N = 1000, solis = 0.005.
diagonal, off_diagonal = morse_diagonals(
    -5.0, 5.0, 0.005
)

energies_7 = eigh_tridiagonal(
    diagonal,
    off_diagonal,
    eigvals_only=True,
    select="i",
    select_range=(0, 4)
)

n = np.arange(5)
analytic_5 = n + 0.5 - (n + 0.5)**2 / (2 * lam)
errors_7 = np.abs(energies_7 - analytic_5)

print("\nSalīdzinājums ar analītisko atrisinājumu")
print("n    7. uzdevums    Analītiskā enerģija    Absolūtā kļūda")

for j in n:
    print(
        f"{j}    {energies_7[j]:.8f}    "
        f"{analytic_5[j]:.8f}    {errors_7[j]:.3e}"
    )
