import numpy as np
import matplotlib.pyplot as plt


def create_hamiltonian(N, xi_max=5.0):
    h = xi_max / N
    xi_inner = np.arange(-N + 1, N) * h

    diagonal = 1.0 / h**2 + xi_inner**2 / 2.0
    off_diagonal = np.full(
        len(xi_inner) - 1,
        -1.0 / (2.0 * h**2)
    )

    H = (
        np.diag(diagonal)
        + np.diag(off_diagonal, k=1)
        + np.diag(off_diagonal, k=-1)
    )

    return H


def current_time():
    return np.datetime64("now", "s")


def elapsed_seconds(start):
    return float(
        (current_time() - start) / np.timedelta64(1, "s")
    )


def measure_diagonalization(
    H,
    minimum_seconds=10.0,
    minimum_repetitions=10
):
    repetitions = 0
    elapsed = 0.0
    start = current_time()

    # Turpinām, kamēr kaut viens nosacījums vēl nav izpildīts.
    while (
        elapsed < minimum_seconds
        or repetitions < minimum_repetitions
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        repetitions += 1
        elapsed = elapsed_seconds(start)

    average_time = elapsed / repetitions

    return average_time, repetitions


# Aprēķina parametri.
xi_max = 5.0
N_values = np.array([50, 100, 150, 250, 500, 1000])

# Katram N veicam trīs atsevišķus mērījumus.
number_of_measurements = 3

# Katrs mērījums ilgst vismaz 10 sekundes
# un ietver vismaz 10 diagonalizācijas.
minimum_seconds = 10.0
minimum_repetitions = 20

times = np.zeros(
    (len(N_values), number_of_measurements)
)

print("Mērām np.linalg.eigh(H) izpildes laiku atkārtojumu ciklā.")
print("Matricas izveides laiks nav iekļauts.\n")

for index, N in enumerate(N_values):
    H = create_hamiltonian(int(N), xi_max)
    M = H.shape[0]

    # Iesildīšanas aprēķins; tā laiku neiekļaujam rezultātā.
    np.linalg.eigh(H)

    print(f"N = {N}, matricas izmērs = {M} x {M}")

    for measurement in range(number_of_measurements):
        average_time, repetitions = measure_diagonalization(
            H,
            minimum_seconds=minimum_seconds,
            minimum_repetitions=minimum_repetitions
        )

        times[index, measurement] = average_time

        print(
            f"  Mērījums {measurement + 1}: "
            f"{average_time:.6f} s vienai diagonalizācijai, "
            f"atkārtojumi: {repetitions}"
        )


# Katram N izmantojam trīs mērījumu mediānu.
median_times = np.median(times, axis=1)

print("\nRezultāti")
print("N     M = 2N - 1     Laiks (s)")

for N, measured_time in zip(N_values, median_times):
    print(
        f"{N:<5} {2*N - 1:<14} {measured_time:.6f}"
    )


# Pēdējiem četriem punktiem pielāgojam t = C * N^p.
fit_count = 4
N_fit = N_values[-fit_count:].astype(float)
time_fit = median_times[-fit_count:]

p, log_C = np.polyfit(
    np.log(N_fit),
    np.log(time_fit),
    1
)

C = np.exp(log_C)

print(
    f"\nPēdējiem {fit_count} punktiem pielāgots t = C * N^p:"
)
print(f"p = {p:.3f}")
print(f"C = {C:.6e}")


# Pielāgojuma līkni attēlojam tikai izmantotajā diapazonā.
N_curve = np.linspace(
    float(N_fit[0]),
    float(N_fit[-1]),
    300
)

fitted_curve = C * N_curve**p

# N^3 salīdzinājuma līkni pieskaņojam pēdējam punktam.
cubic_C = median_times[-1] / float(N_values[-1])**3
cubic_curve = cubic_C * N_curve**3


# 1. Atsevišķs grafiks parastajā skalā.
fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    N_values,
    median_times,
    "o-",
    label="Izmērītais laiks"
)

ax1.set_xlabel(r"$N$")
ax1.set_ylabel("Vienas diagonalizācijas laiks (s; lin)")
ax1.grid()
ax1.legend()

fig1.tight_layout()


# 2. Atsevišķs grafiks logaritmiskajā skalā.
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.loglog(
    N_values,
    median_times,
    "o",
    label="Izmērītais laiks"
)

ax2.loglog(
    N_curve,
    fitted_curve,
    "--",
    label=rf"Pielāgojums: $t \propto N^{{{p:.2f}}}$"
)

ax2.loglog(
    N_curve,
    cubic_curve,
    ":",
    label=r"Salīdzinājums: $t \propto N^3$"
)

ax2.set_xlabel(r"$N$")
ax2.set_ylabel("Vienas diagonalizācijas laiks (s; log)")
ax2.grid()
ax2.legend()

fig2.tight_layout()

plt.show()