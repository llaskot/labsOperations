import numpy as np
import matplotlib.pyplot as plt


def build_3d_graph_swen(x_points, fx_points, a, b, c, func_lamb, func_str="f(x)"):
    n = len(x_points)
    k = list(range(n - 1, -1, -1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)

    x_min, x_max = min(x_points), max(x_points)
    tol = (x_max - x_min) / 5
    x_line = np.linspace(x_min - tol, x_max + tol, 250)

    # рисуем графики и точки
    for i, ki in enumerate(k):
        ax.plot(x_line, func_lamb(x_line), np.full_like(x_line, ki),
                color='black', linestyle='--', linewidth=1)
        ax.scatter(x_points[i], fx_points[i], ki, color='red', s=20)

    ax.plot(x_points, fx_points, k, color='red', linewidth=1)

    # подготовка для специальных точек
    pts = [a, b, c]
    x_to_k = {x: ki for x, ki in zip(x_points, k)}
    for pt in pts:
        ax.scatter(pt[0], pt[1], x_to_k[pt[0]], color='blue', s=30)

    # соединение первых двух специальных точек
    k_pts = [x_to_k[a[0]], x_to_k[b[0]]]
    ax.plot([a[0], b[0]], [a[1], b[1]], k_pts, color='green', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('f(X)')
    ax.set_zlabel('k')
    ax.set_zticks(k)
    ax.set_zticklabels(reversed(k))

    ax.plot([], [], color='black', linestyle='--', linewidth=1, label=func_str)
    ax.plot([], [], color='green', linewidth=2, label="uncertainty interval (a:b)")
    ax.legend()

    plt.show()
