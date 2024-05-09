import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_spin_lattice(spins, title, cmap='binary'):
    """Funkcja do rysowania siatki i wykresu słupkowego."""
    plt.figure(figsize=(8, 8))

    # Wyświetlanie siatki spinów
    plt.imshow(spins, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

    # Znalezienie unikalnych wartości spinów i ich liczności
    unique, counts = np.unique(spins, return_counts=True)


    if cmap == 'binary':
        # Dla modelu Isinga
        colors = {-1: 'grey', 1: 'black'}
    else:
        # Dla modelu Potts-Dirichleta
        # Losowanie kolorów dla każdej wartości spinu i zapisanie ich do słownika
        hsv_colors = plt.cm.hsv(np.linspace(0, 1, len(unique)))
        colors = dict(zip(unique, hsv_colors))

    # Rysowanie wykresu słupkowego odpowiadającego liczbie wystąpień danej wartości spinu
    for i, (spin, count) in enumerate(zip(unique, counts)):
        color = colors[spin] if cmap != 'binary' else colors.get(spin, 'grey')
        plt.barh(i, count, color=color)

    # Ustawienie etykiet osi y i ich wartości
    plt.yticks(range(len(unique)), unique)
    plt.gca().invert_yaxis()  # Odwrócenie osi y, aby etykiety były od dołu do góry
    plt.xlabel('Liczba wystąpień')
    plt.ylabel('Wartość spinu')
    plt.title('Rozkład spinów')
    plt.show()


class IsingModel:
    """Klasa reprezentująca model Isinga."""

    def __init__(self, size, temperature):
        """Inicjalizacja modelu."""
        self.size = size
        self.temperature = temperature
        self.spins = np.random.choice([-1, 1], size=(size, size))

    def energy(self):
        """Obliczenie energii."""
        return -np.sum(self.spins * (np.roll(self.spins, 1, axis=0) + np.roll(self.spins, 1, axis=1)))

    def magnetization(self):
        """Obliczenie magnetyzacji."""
        return np.sum(self.spins)

    def monte_carlo_step(self):
        """Jeden krok Monte Carlo."""
        i, j = np.random.randint(0, self.size, 2)
        dE = 2 * self.spins[i, j] * (self.spins[(i + 1) % self.size, j] + self.spins[(i - 1) % self.size, j] +
                                     self.spins[i, (j + 1) % self.size] + self.spins[i, (j - 1) % self.size])
        if dE <= 0 or np.random.rand() < np.exp(-dE / self.temperature):
            self.spins[i, j] *= -1


class PottsDirichletModel:
    """Klasa reprezentująca model Potts-Dirichleta."""

    def __init__(self, size, temperature, q):
        """Inicjalizacja modelu."""
        self.size = size
        self.temperature = temperature
        self.q = q
        self.spins = np.random.randint(1, q + 1, size=(size, size))

    def energy(self):
        """Obliczenie energii."""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                energy -= sum(
                    self.spins[i, j] == self.spins[(i + di) % self.size, (j + dj) % self.size] for di in [-1, 1] for dj
                    in [-1, 1])
        return energy

    def magnetization(self):
        """Obliczenie magnetyzacji."""
        return np.sum(self.spins)

    def monte_carlo_step(self):
        """Jeden krok Monte Carlo."""
        i, j = np.random.randint(0, self.size, 2)
        old_spin = self.spins[i, j]
        new_spin = old_spin
        while new_spin == old_spin:
            new_spin = np.random.randint(1, self.q + 1)
        dE = sum(new_spin == self.spins[(i + di) % self.size, (j + dj) % self.size] -
                 old_spin == self.spins[(i + di) % self.size, (j + dj) % self.size] for di in [-1, 1] for dj in [-1, 1])
        if dE <= 0 or np.random.rand() < np.exp(-dE / self.temperature):
            self.spins[i, j] = new_spin


def plot_spin_matrix(spins, title):
    """Funkcja rysuje macierz spinów w postaci siatki 2D."""
    plt.imshow(spins, cmap='gray', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()


def calculate_energy(spins, J):
    """
    Funkcja calculate_energy(spins, J) oblicza energię układu spinów
    w modelu Isinga.
    spins: macierz spinów
    J: stała sprzężenia magnetycznego
    """
    energy = 0
    N = spins.shape[0]
    for i in range(N):
        for j in range(N):
            energy += -J * spins[i, j] * (spins[(i + 1) % N, j] + spins[i, (j + 1) % N])
    return energy


def metropolis(spins, J, kT, num_steps):
    """
    Algorytm Metropolisa do symulacji modelu Isinga.
    spins: macierz spinów
    J: stała sprzężenia magnetycznego
    kT: iloczyn stałej Boltzmanna i temperatury
    num_steps: liczba kroków symulacji
    """
    N = spins.shape[0]
    energies = []
    for step in range(num_steps):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        spin_flip_energy = 2 * J * spins[i, j] * (spins[(i - 1) % N, j] + spins[(i + 1) % N, j] +
                                                  spins[i, (j - 1) % N] + spins[i, (j + 1) % N])
        if spin_flip_energy <= 0 or np.exp(-spin_flip_energy / kT) > np.random.rand():
            spins[i, j] *= -1
        energy = calculate_energy(spins, J)
        energies.append(energy)
    return spins, energies




def simulate_ising_and_potts(size, temperature, q):
    # Inicjalizacja modelu Isinga
    ising_model = IsingModel(size, temperature)
    ising_energies = []  # Lista przechowująca energie
    ising_magnetizations = []  # Lista przechowująca magnetyzacje
    ising_spins_before = np.copy(ising_model.spins)  # Przed zmianą
    for _ in range(1000):
        ising_model.monte_carlo_step()  # Jeden krok Monte Carlo
        ising_energies.append(ising_model.energy())  # Dodanie energii do listy
        ising_magnetizations.append(ising_model.magnetization())  # Dodanie magnetyzacji do listy
    ising_spins_after = np.copy(ising_model.spins)  # Po zmianie

    # Inicjalizacja modelu Potts-Dirichleta
    potts_model = PottsDirichletModel(size, temperature, q)
    potts_energies = []  # Lista przechowująca energie
    potts_magnetizations = []  # Lista przechowująca magnetyzacje
    potts_spins_before = np.copy(potts_model.spins)  # Przed zmianą
    for _ in range(1000):
        potts_model.monte_carlo_step()  # Jeden krok Monte Carlo
        potts_energies.append(potts_model.energy())  # Dodanie energii do listy
        potts_magnetizations.append(potts_model.magnetization())  # Dodanie magnetyzacji do listy
    potts_spins_after = np.copy(potts_model.spins)  # Po zmianie

    # Wyświetlanie siatek przed i po zmianach
    plot_spin_lattice(ising_spins_before, "Siatka Isinga przed zmianą")
    plot_spin_lattice(ising_spins_after, "Siatka Isinga po zmianie")
    plot_spin_lattice(potts_spins_before, "Siatka Potts-Dirichleta przed zmianą", cmap='hsv')
    plot_spin_lattice(potts_spins_after, "Siatka Potts-Dirichleta po zmianie", cmap='hsv')

    # Wykresy zmiany energii i magnetyzacji
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ising_energies, label='Ising Model')
    plt.plot(potts_energies, label='Potts Model (q={})'.format(q))
    plt.title('Zmiana Energii')
    plt.xlabel('Kroki Monte Carlo')
    plt.ylabel('Energia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ising_magnetizations, label='Ising Model')
    plt.plot(potts_magnetizations, label='Potts Model (q={})'.format(q))
    plt.title('Zmiana Magnetyzacji')
    plt.xlabel('Kroki Monte Carlo')
    plt.ylabel('Magnetyzacja')
    plt.legend()  # Dodanie legendy

    plt.tight_layout()  # Dopasowanie układu wykresów
    plt.show()  # Wyświetlenie wykresów

    return ising_energies, potts_energies, ising_magnetizations, potts_magnetizations

# Wczytanie parametrów od użytkownika
size = int(input("Podaj rozmiar siatki: ")) # Pobranie rozmiaru siatki od użytkownika
temperature = float(input("Podaj temperaturę: ")) # Pobranie temperatury od użytkownika
q = int(input("Podaj parametr q: ")) # Pobranie parametru q od użytkownika

# Wywołanie funkcji symulacji
simulate_ising_and_potts(size, temperature, q)


# Implementacja sugestii

def analyze_phase_transitions(sizes=[10, 20, 30]):
    """Funkcja analizuje przejścia fazowe dla różnych rozmiarów siatki."""
    critical_temperatures = []  # Lista przechowująca krytyczne temperatury dla modelu Isinga

    # 1. Analiza przejść fazowych
    for size in sizes:
        ising_energies = []  # Lista przechowująca energie dla danego rozmiaru siatki
        for temperature in np.linspace(0.5, 5.0, 50):
            ising_model = IsingModel(size, temperature)  # Inicjalizacja modelu Isinga
            for _ in range(1000):
                ising_model.monte_carlo_step()  # Jeden krok Monte Carlo
            ising_energies.append(np.mean([ising_model.energy() for _ in range(1000)]))  # Obliczenie średniej energii
        # Dodanie danych do wykresu
        plt.plot(np.linspace(0.5, 5.0, 50), ising_energies, label='Size {}'.format(size))
        # Obliczenie krytycznej temperatury
        critical_temperature = float(np.linspace(0.5, 5.0, 50)[np.argmin(ising_energies)])
        critical_temperatures.append(critical_temperature)  # Dodanie krytycznej temperatury do listy

    plt.title('Analiza Przejść Fazowych (Model Isinga)')  # Ustawienie tytułu wykresu
    plt.xlabel('Temperatura')  # Oznaczenie osi x
    plt.ylabel('Energia')  # Oznaczenie osi y
    plt.legend()  # Dodanie legendy
    plt.show()  # Wyświetlenie wykresu

    print("Krytyczne temperatury dla modelu Isinga:", critical_temperatures)  # Wyświetlenie krytycznych temperatur

    return critical_temperatures

def find_clusters(spins):
    """Funkcja znajduje klastry spinów."""
    visited = np.zeros_like(spins)  # Utworzenie macierzy odwiedzonych pól
    clusters = []  # Lista przechowująca klastry

    def dfs(i, j, cluster):
        """Przeszukiwanie w głąb dla znalezienia klastra."""
        if visited[i, j] == 1 or spins[i, j] != 1:  # Warunek zakończenia rekurencji
            return
        visited[i, j] = 1  # Oznaczenie pola jako odwiedzone
        cluster.append((i, j))  # Dodanie pola do klastra
        for ni, nj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:  # Iteracja po sąsiadach
            if 0 <= ni < spins.shape[0] and 0 <= nj < spins.shape[1]:  # Sprawdzenie czy sąsiad mieści się w siatce
                dfs(ni, nj, cluster)  # Wywołanie rekurencyjne dla sąsiada

    for i in range(spins.shape[0]):  # Iteracja po rzędach siatki
        for j in range(spins.shape[1]):  # Iteracja po kolumnach siatki
            if spins[i, j] == 1 and visited[i, j] == 0:  # Warunek na znalezienie nieodwiedzonego pola z spinem
                cluster = []  # Inicjalizacja klastra
                dfs(i, j, cluster)  # Wywołanie funkcji przeszukiwania w głąb
                if cluster:  # Jeśli znaleziono jakiś klastr, to dodaj go do listy
                    clusters.append(cluster)

    return clusters  # Zwrócenie listy klastrów

def plot_cluster_size_distribution(size=30, temperatures=[1.0, 2.0, 3.0]):
    """Funkcja bada rozkład rozmiarów klastrów spinów."""
    for temperature in temperatures:
        ising_model = IsingModel(size, temperature)  # Inicjalizacja modelu Isinga
        clusters = []  # Lista przechowująca klastry

        for _ in range(1000):
            ising_model.monte_carlo_step()  # Jeden krok Monte Carlo
            clusters.append(find_clusters(ising_model.spins))  # Dodanie klastrów do listy

        cluster_sizes = [len(cluster) for cluster in clusters]  # Lista przechowująca rozmiary klastrów
        plt.hist(cluster_sizes, bins=range(min(cluster_sizes), max(cluster_sizes) + 1, 1), density=True, alpha=0.5,
                 label='Temperature = {}'.format(temperature))  # Tworzenie histogramu rozmiarów klastrów

    plt.title('Rozkład Rozmiarów Klastrów Spinów (Model Isinga, Size = {})'.format(size))  # Ustawienie tytułu wykresu
    plt.xlabel('Rozmiar klastra')  # Oznaczenie osi x
    plt.ylabel('Częstość')  # Oznaczenie osi y
    plt.legend()  # Dodanie legendy
    plt.show()  # Wyświetlenie wykresu

def plot_3d_energy_surface(energies, title):
    """Funkcja rysuje powierzchnię trójwymiarową energii w zależności od kroków i temperatury."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    steps = len(energies[0])
    temperatures = np.linspace(0.5, 5.0, len(energies))
    X, Y = np.meshgrid(range(steps), temperatures)
    Z = np.array(energies)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Kroki Monte Carlo')
    ax.set_ylabel('Temperatura')
    ax.set_zlabel('Energia')
    ax.set_title(title)
    plt.show()

# Przykładowe wywołanie funkcji
analyze_phase_transitions()
plot_cluster_size_distribution()

# Przykładowe użycie

size = 10
temperature = 2.0
q = 3

# Symulacja modelu Isinga
ising_model = IsingModel(size, temperature)
ising_energies = []

for _ in range(50):
    ising_model.monte_carlo_step()
    ising_energies.append([ising_model.energy()] * 1000)

plot_3d_energy_surface(ising_energies, "Zmiana Energii w Modelu Isinga")

# Symulacja modelu Potts-Dirichleta
potts_model = PottsDirichletModel(size, temperature, q)
potts_energies = []

for _ in range(50):
    potts_model.monte_carlo_step()
    potts_energies.append([potts_model.energy()] * 1000)

plot_3d_energy_surface(potts_energies, "Zmiana Energii w Modelu Potts-Dirichleta")

