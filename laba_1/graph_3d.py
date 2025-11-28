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


def build_3d_decreasing(x_points, fx_points, func_lamb, func_str="f(x)"):
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

    ax.set_xlabel('X')
    ax.set_ylabel('f(X)')
    ax.set_zlabel('k')
    ax.set_zticks(k)
    ax.set_zticklabels(reversed(k))

    ax.plot([], [], color='black', linestyle='--', linewidth=1, label=func_str)
    ax.plot([], [], color='red', linewidth=1, label="path to success")
    ax.legend(
        loc='upper center',  # Центрирование по горизонтали
        bbox_to_anchor=(0.1, 1.15),  # Смещение в точку (0.5, 1.15) относительно осей
        ncol=2  # Размещение элементов в 2 колонки
    )
    plt.show()




def plot_3d_surface_with_path(func, x_points, fx_points, color_grad = False, z_margin_percent=None, func_str="f(x, y)"):
    """
    func: функция f(np.array([x, y])) -> float
    x_points: список np.array([x, y])
    fx_points: список значений функции в этих точках
    """
    # Преобразуем точки в координаты
    x_arr = np.array([p[0] for p in x_points])
    y_arr = np.array([p[1] for p in x_points])
    # z_arr = np.array(fx_points)
    z_arr = fx_points

    # отступы от краев
    padding_x = (x_arr.max() - x_arr.min()) * 0.2 + 1e-3
    padding_y = (y_arr.max() - y_arr.min()) * 0.2 + 1e-3

    # размах / Определение координат сетки
    X = np.linspace(x_arr.min() - padding_x, x_arr.max() + padding_x, 200)  # Использует padding_x
    Y = np.linspace(y_arr.min() - padding_y, y_arr.max() + padding_y, 200)  # Использует padding_y
    X, Y = np.meshgrid(X, Y)

    # Векторизуем функцию для сетки
    func_lamb = np.vectorize(lambda x, y: func(np.array([x, y])))
    Z = func_lamb(X, Y)

    if z_margin_percent:
        z_path_max = max(z_arr)
        z_path_min = min(z_arr)

        # Допуск (маржа) для обрезки
        z_margin = (z_path_max - z_path_min) * (z_margin_percent / 100.0)

        # Установка пределов оси Z (Z_lim)
        z_lim_max = z_path_max + z_margin
        z_lim_min = z_path_min - z_margin

        Z[Z > z_lim_max] = z_lim_max
        Z[Z < z_lim_min] = z_lim_min

    # Построение графика
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(
        left=0.01,  # Минимальное поле слева (ближе к 0)
        right=0.99,  # Минимальное поле справа (ближе к 1)
        bottom=0.01,  # Минимальное поле снизу
        top=0.99,  # Минимальное поле сверху
        wspace=0.0,  # Горизонтальный интервал между графиками (здесь не нужен, т.к. только 1 график)
        hspace=0.0  # Вертикальный интервал между графиками
    )

    # Поверхность (чаша)
    ax.plot_surface(X, Y, Z, alpha=0.4, cmap='Spectral', rstride=3, cstride=3)

    # Схема
    # Параметр
    # cmap
    # Описание
    # Viridis
    # viridis
    # '	Рекомендуемая по умолчанию. Создана для лучшего восприятия и различия даже при дальтонизме. От темно-фиолетового до ярко-желтого.
    # Plasma
    # plasma
    # '	Яркая, с интенсивным свечением, хорошо подходит для визуализации быстро растущих данных.
    # Magma
    # magma
    # '	От темного (почти черного) к светло-оранжевому/белому, часто используется для отображения тепловых карт.
    # Inferno
    # inferno
    # '	Схожа с Magma, но с более выраженными красными и оранжевыми оттенками.
    # Blues
    # Blues
    # '	Плавный переход от светлого к насыщенному синему.
    # Greens
    # Greens
    # '	Плавный переход от светлого к насыщенному зеленому.
    # Coolwarm
    # coolwarm
    # '	Идеальна для нуля. Переход от синего (отрицательные значения) через белый (ноль) к красному (положительные значения).
    # RdBu
    # RdBu
    # '	Классическая схема Red-Blue, часто используется для отображения отклонений от среднего.
    # Rainbow
    # rainbow
    # '	Использует весь спектр цветов; полезна для циклических данных, но иногда затрудняет восприятие последовательности.
    # Jet
    # jet
    # '	Классическая и часто используемая, но не всегда лучшая для научного восприятия.
    # Tab10
    # tab10
    # '	Набор дискретных, контрастных цветов, полезен, если нужно различать отдельные категории, а не плавный переход.

    # Линия спуска



    # ax.plot(x_arr, y_arr, z_arr, color='red', marker='o', linewidth=0.5, label="path to minimum", markersize=2)

    if color_grad:
        # 1. Сначала убираем старую линию
        # (или заменяем ее на более тонкую, если она нужна)
        ax.plot(x_arr, y_arr, z_arr, color='gray', linewidth=0.5)

        # 2. Используем ax.scatter
        # color='red' заменяем на массив z_arr
        # cmap='viridis' - это палитра цветов (colormap), которая будет применена к z_arr
        path = ax.scatter(
            x_arr,
            y_arr,
            z_arr,
            c=z_arr,  # <— Значения Z определяют цвет
            cmap='jet',  # <— Выберите подходящую палитру
            marker='o',
            label="path to minimum",
            s=15  # Размер точек
        )

        # 3. Добавьте цветовую шкалу (colorbar), чтобы понять, какой цвет какому значению соответствует
        fig.colorbar(path, ax=ax, label=f'{func_str} value')
    else:
        ax.plot(x_arr, y_arr, z_arr, color='red', marker='o', linewidth=0.5, label="path to minimum", markersize=2)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(func_str)
    ax.legend()

    # plt.show()
    # plt.close(fig)

    return fig

