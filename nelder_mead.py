
import numpy as np
import matplotlib.pyplot as plt


def f(x1, x2):
    """
       Funkcja celu: f(x1, x2) = x1^2 + x2^2

       Args:
           x1 (float or np.ndarray): Pierwsza zmienna wejściowa.
           x2 (float or np.ndarray): Druga zmienna wejściowa.

       Returns:
            Wartość funkcji celu w punkcie (x1, x2).
       """
    return np.power(x1, 2) + np.power(x2, 2)


def nelder_mead(f, x_start, x_end, s, alpha, beta, gamma, delta, epsilon):
    """
        Implementacja algorytmu Neldera-Meada dla optymalizacji funkcji celu.

        Args:
            f (function): Funkcja celu.
            x_start (float): Początkowy punkt dla zmiennej x1.
            x_end (float): Początkowy punkt dla zmiennej x2.
            s (float): Skala inicjalizacji wierzchołków sympleksu.
            alpha (float): Współczynnik odbicia.
            beta (float): Współczynnik zwężania.
            gamma (float): Współczynnik ekspansji.
            delta (float): Współczynnik redukcji.
            epsilon (float): Kryterium zakończenia (tolerancja).

        Returns:
            Punkt minimum funkcji celu.
        """

    p0 = np.array([x_start, x_end])
    n = len(p0)
    f_array = [f(p0[0], p0[1])]
    p = [p0]

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        new_vertex = p0 + s * e_i
        p.append(new_vertex)
        f_array.append(f(new_vertex[0], new_vertex[1]))

    p = np.array(p)
    p_min = min(p, key=lambda vertex: f(vertex[0], vertex[1]))
    p_max = max(p, key=lambda vertex: f(vertex[0], vertex[1]))
    p_ = 0.5 * (p[0] + p[1])
    p_odb = p_ + alpha * (p_ - p_max)

    while True:
        if f(p_odb[0], p_odb[1]) < f(p_min[0], p_min[1]):
            p_e = p_ + gamma * (p_odb - p_)
            if f(p_e[0], p_e[1]) < f(p_odb[0], p_odb[1]):
                p_max = p_e
            else:
                p_max = p_odb
        else:
            if f(p_odb[0], p_odb[1]) >= f(p_min[0], p_min[1]) and f(p_odb[0], p_odb[1]) < f(p_max[0], p_max[1]):
                p_max = p_odb
            else:
                p_z = p_ + beta * (p_max - p_)
                if f(p_z[0], p_z[1]) >= f(p_max[0], p_max[1]):
                    for i in range(len(p)):
                        p[i] = delta * (p[i] + p_min)
                else:
                    p_max = p_z

        if max([np.linalg.norm(p_min - p[i]) for i in range(len(p))]) < epsilon:
            x_star = p_min
            break

    return x_star


def visualize_optimization(f, resolution=100):
    """
    Wizualizacja funkcji celu w przestrzeni 3D.

    """
    x = np.linspace(-4, 4, resolution)
    y = np.linspace(-4, 4, resolution)
    x, y = np.meshgrid(x, y)
    z = f(x, y)


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='plasma', alpha=0.8)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.title('Funkcja celu: $f(x_1, x_2) = x_1^2 + x_2^2$')
    plt.show()


if __name__ == "__main__":

    x_start = -0.5
    x_end = 0.5
    s = 1.0
    alpha = 1.0
    beta = 0.5
    gamma = 2.0
    delta = 0.5
    epsilon = 0.01


    minimum = nelder_mead(f, x_start, x_end, s, alpha, beta, gamma, delta, epsilon)
    print("Minimum znalezione przez algorytm:", minimum)


    visualize_optimization(f)


