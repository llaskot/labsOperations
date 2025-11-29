from math import sqrt

import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

from laba_1.graph_3d import plot_3d_surface_with_path


class Koshi:
    # def __init__(self, func_str: str = "4*x_1**2 + x_2**2", dim: int = 2, n_max: int = 300):

    # def __init__(self, func_str: str = "(1 - x_1)**2 + 100*(x_2 - x_1**2)**2", dim: int = 2, n_max: int = 5000):
    def __init__(self, func_str: str = "x_1**2 + (x_2 - 2)**2", dim: int = 2, n_max: int = 5000):

        self.fc = None
        self.fb = None
        self.fa = None
        self.c = None
        self.b = None
        self.a = None
        self.func_str = func_str
        self.dim = dim
        self.func = self._create_func_wrapper()
        self.n_max = n_max

        r_final = 1
        self.r_0 = 1
        self.all_x = []
        self.all_fx = []
        self.n_final = -1

    def _create_func_wrapper(self):
        symbols_list = [sp.symbols(f'x_{i + 1}') for i in range(self.dim)]
        f_sym = sp.sympify(self.func_str)
        f_lambda_base = sp.lambdify(symbols_list, f_sym, 'numpy')

        def func_wrapper(x_vec: np.ndarray) -> float:
            return f_lambda_base(*x_vec)

        return func_wrapper

    def get_val(self, x_vect: np.ndarray) -> float:
        return self.func(x_vect)

    def calculate_gradient(self, x_vec: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        g_vec = np.zeros(self.dim)
        x_vec = np.array(x_vec, dtype=float)
        fx_val = self.func(x_vec)
        y_vec = np.copy(x_vec)
        for i in range(self.dim):
            y_vec[i] = float(y_vec[i]) + delta
            fy_val = self.func(y_vec)
            g_vec[i] = (fy_val - fx_val) / delta
            y_vec[i] = x_vec[i]
        return g_vec

    def swen_method(self, g_vec: np.ndarray, x0: tuple = (5.0, 5.0), lambda_0: float = 1, max_steps: int = 100):
        x_base = np.array(x0)
        cur_step = 0
        h = lambda_0
        c = 0.0

        def y(lambda_val: float):
            return x_base - lambda_val * g_vec

        fc = self.get_val(y(c))
        b = c + h
        cur_step += 1
        fb = self.get_val(y(b))
        if fb >= fc:
            # self.a, self.b, = c, b
            # self.fa, self.fb = fc, fb
            a = c - h
            cur_step += 1
            fa = self.get_val(y(a))
            if fa >= fc:
                self.a, self.b, self.c = a, b, c
                self.fa, self.fb, self.fc = fa, fb, fc
            else:
                self._swen_inc(b, fb, c, fc, h, max_steps, cur_step, y)
        else:
            self._swen_dec(c, fc, b, fb, h, max_steps, cur_step, y)

    def _swen_dec(self, a, fa, c, fc, h, max_steps, cur_step, y):
        while cur_step < max_steps:
            h *= 2
            b = c + h
            cur_step += 1
            fb = self.get_val(y(b))
            if fb < fc:
                a, fa, c, fc = c, fc, b, fb
            else:
                self.a, self.b, self.c = a, b, c
                self.fa, self.fb, self.fc = fa, fb, fc
                return

    def _swen_inc(self, b, fb, c, fc, h, max_steps, cur_step, y):
        while cur_step < max_steps:
            h *= 2
            a = c - h
            cur_step += 1
            fa = self.get_val(y(a))
            if fa >= fc:
                self.a, self.b, self.c = a, b, c
                self.fa, self.fb, self.fc = fa, fb, fc
                return
            else:
                b, fb, c, fc = c, fc, a, fa

    def corop_line(self, x0: np.ndarray, g_vec: np.ndarray, tol: float) -> float:
        cur_step = -1
        h = tol * 10
        x_base = np.copy(x0)
        final = 0
        x = 0.0
        r = 0

        def x_vec(lambda_val: float):
            return x_base - lambda_val * g_vec

        cur_step += 1
        fx = self.get_val(x_vec(x))
        print(f"\n{fx=}")

        while cur_step < self.n_max:
            y = x + h
            cur_step += 1
            fy = self.get_val(x_vec(y))
            print(f"\n{fy=}")
            print("step = ", h)
            print("r inside = ", r)
            if fy >= fx:
                print("wrong way ")
                if r == 2:
                    r = 0.25
                else:
                    r = - 0.5
            else:
                final = y
                x = y
                fx = fy
                if r >= 0.5:
                    r = 2
                else:
                    r = 0.5
            h = h * r
            if not abs(h) > tol:
                return final

    def quick_down(self, x0: tuple, tol: float):
        x_vec = np.array(x0)
        result = []
        result_func = []
        k = 0
        while k < self.n_max:
            k += 1
            result.append(x_vec)
            result_func.append(self.get_val(x_vec))
            g_vec = self.calculate_gradient(x_vec)

            # r = self.corop_line(x_vec, g_vec, tol * 10)
            self.swen_method(g_vec, x_vec, 0.01)
            r = self.binary(self.a, self.b, g_vec, x_vec, tol * 10)
            # r = self.dichotomy(self.a, self.b, g_vec, x_vec, tol * 10)
            # r = self.gold(self.a, self.b, g_vec, x_vec, tol * 10)

            print(f"{r=}")
            s = -r * g_vec
            x_vec = x_vec + s
            print("s scalar = ", np.linalg.norm(s))
            print("\n\n")
            if np.linalg.norm(s) <= tol:
                return result, result_func
        return result, result_func

    def polocoribyer(self, x0: tuple, tol: float):
        x_vec = np.array(x0)
        result = []
        result_func = []
        k = 0
        g_vec_x = self.calculate_gradient(x_vec)
        d_vec = -g_vec_x
        while k < self.n_max:
            k += 1
            result.append(x_vec)
            result_func.append(self.get_val(x_vec))
            # r = self.corop_line(x_vec, -d_vec, tol * 10)
            self.swen_method(-d_vec, x_vec, 0.01)
            # r = self.binary(self.a, self.b, -d_vec, x_vec, tol * 10)
            # r = self.dichotomy(self.a, self.b, -d_vec, x_vec, tol * 10)
            r = self.gold(self.a, self.b, -d_vec, x_vec, tol * 10)

            print(f"{r=}")
            s = r * d_vec
            x_vec = x_vec + s
            g_vec_y = np.copy(g_vec_x)
            g_vec_x = self.calculate_gradient(x_vec)
            beta = (g_vec_x @ (g_vec_x - g_vec_y)) / (g_vec_y @ g_vec_y)
            d_vec = -g_vec_x + beta * d_vec
            print("s scalar = ", np.linalg.norm(s))
            print("\n\n")
            if np.linalg.norm(s) <= tol:
                return result, result_func
        return result, result_func

    def fl_rivs(self, x0: tuple, tol: float):
        x_vec = np.array(x0)
        result = []
        result_func = []
        k = 0
        g_vec_x = self.calculate_gradient(x_vec)
        d_vec = -g_vec_x
        while k < self.n_max:
            k += 1
            result.append(x_vec)
            result_func.append(self.get_val(x_vec))
            # r = self.corop_line(x_vec, -d_vec, tol*10)
            self.swen_method(-d_vec, x_vec, 0.01)
            # r = self.binary(self.a, self.b, -d_vec, x_vec, tol * 10)
            # r = self.dichotomy(self.a, self.b, -d_vec, x_vec, tol * 10)
            r = self.gold(self.a, self.b, -d_vec, x_vec, tol * 10)

            print(f"{r=}")
            s = r * d_vec
            x_vec = x_vec + s
            g_vec_y = np.copy(g_vec_x)
            g_vec_x = self.calculate_gradient(x_vec)
            print(f"{g_vec_x}")
            print('g_vec_x @ g_vec_x = ', g_vec_x @ g_vec_x)
            if g_vec_x @ g_vec_x == 0:
                break
            beta = (g_vec_x @ g_vec_x) / (g_vec_y @ g_vec_y)

            print(f"{beta=}")
            d_vec = -g_vec_x + beta * d_vec
            print("s scalar = ", np.linalg.norm(s))
            print('tolerance = ', tol)
            print("\n\n")
            if np.linalg.norm(s) <= tol:
                return result, result_func
        return result, result_func

    def binary(self, a: float, b: float, g_vec: np.ndarray, x_base: np.ndarray, tol: float) -> float:
        def y(lambda_val: float):
            return x_base - lambda_val * g_vec

        c = (a + b) / 2
        fc = self.get_val(y(c))
        while True and self.n_final < self.n_max:
            u = (a + c) / 2
            fu = self.get_val(y(u))
            if fu < fc:
                b = c
                c = u
                fc = fu
            else:
                v = (c + b) / 2
                fv = self.get_val(y(v))
                if fv < fc:
                    a = c
                    c = v
                    fc = fv
                else:
                    a = u
                    b = v
            print(f'uncertainty interval = {b - a}')
            if (b - a) <= tol:
                return (b + a) / 2

    def dichotomy(self, a: float, b: float, g_vec: np.ndarray, x_base: np.ndarray, tol: float) -> float:
        def y(lambda_val: float):
            return x_base - lambda_val * g_vec

        shift = tol / 10
        while True and self.n_final < self.n_max:
            c = (a + b) / 2
            u = c - shift
            v = c + shift
            fu = self.get_val(y(u))
            fv = self.get_val(y(v))
            if fu < fv:
                b = v
            else:
                a = u
            print(f'uncertainty interval = {b - a}')
            if not b - a > tol:
                return (a + b) / 2

    def gold(self, a: float, b: float, g_vec: np.ndarray, x_base: np.ndarray, tol: float) -> float:
        def x_vec(lambda_val: float):
            return x_base - lambda_val * g_vec

        t = 0.5 * (sqrt(5) - 1)
        h = t * (b - a)
        x = b - h
        fx = self.get_val(x_vec(x))
        y = a + h
        fy = self.get_val(x_vec(y))
        while True and self.n_final < self.n_max:
            if fx > fy:
                a = x
                x = y
                fx = fy
                d = b - a
                y = a + t * d
                fy = self.get_val(x_vec(y))
            else:
                b = y
                y = x
                fy = fx
                d = b - a
                x = b - t * d
                fx = self.get_val(x_vec(x))
            print(f'uncertainty interval = {d}')
            if not d > tol:
                if fy < fx:
                    x = y
                    fx = fy
                return x

    def newthon(self, x0: tuple, tol: float):
        x_vec = np.array(x0)
        result = []
        result_func = []
        k = 0
        while k < self.n_max:
            k += 1
            result.append(x_vec)
            result_func.append(self.get_val(x_vec))
            g_vec = self.calculate_gradient_central(x_vec)
            hessian = self.calculate_hessian(x_vec)
            d_vec = self.find_newton_direction(hessian, g_vec)

            # r = self.corop_line(x_vec, -d_vec, tol * 10)
            self.swen_method(-d_vec, x_vec, 0.01)
            # r = self.binary(self.a, self.b, -d_vec, x_vec, tol * 10)
            # r = self.dichotomy(self.a, self.b, -d_vec, x_vec, tol * 10)
            r = self.gold(self.a, self.b, -d_vec, x_vec, tol * 10)

            print(f"{r=}")
            s = r * d_vec
            x_vec = x_vec + s
            print("s scalar = ", np.linalg.norm(s))
            print("\n\n")
            if np.linalg.norm(s) <= tol:
                return result, result_func
        return result, result_func

    def calculate_gradient_central(self, x_vec: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        g_vec = np.zeros(self.dim)

        for i in range(self.dim):
            x_plus = x_vec.copy()
            x_minus = x_vec.copy()

            # 1. Считаем f(x + delta*e_i)
            x_plus[i] += delta
            fy_val = self.get_val(x_plus)

            # 2. Считаем f(x - delta*e_i)
            x_minus[i] -= delta
            fz_val = self.get_val(x_minus)

            # 3. Применяем формулу центральных разностей
            g_vec[i] = (fy_val - fz_val) / (2 * delta)
        return g_vec

    def calculate_hessian(self, x_vec: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        H = np.zeros((self.dim, self.dim))
        fx = self.func(x_vec)  # f(x)
        d2 = delta ** 2

        # 1. Рассчитываем и сохраняем f(x + delta*e_i) для каждой оси
        # Мы объединяем этот расчет с расчетом H_ii для оптимизации,
        # чтобы не делать лишний цикл для H_ii.
        f_plus_i = np.zeros(self.dim)
        f_minus_i = np.zeros(self.dim)

        # --- 1 & 2. Расчет и сохранение f(x +/- delta*e_i) и H_ii (Диагональные элементы) ---
        for i in range(self.dim):
            # 1a. f(x + delta*e_i)
            x_plus_i = x_vec.copy()
            x_plus_i[i] += delta
            f_plus_i[i] = self.func(x_plus_i)  # Сохраняем f_plus_i[i] для H_ij

            # 1b. f(x - delta*e_i)
            x_minus_i = x_vec.copy()
            x_minus_i[i] -= delta
            f_minus_i_val = self.func(x_minus_i)

            # 2. Расчет Диагонального элемента H_ii (ВНУТРИ ЦИКЛА)
            # H_ii = (f(x + d*e_i) - 2*f(x) + f(x - d*e_i)) / d^2
            H[i, i] = (f_plus_i[i] - 2 * fx + f_minus_i_val) / d2

        # --- 3. Расчет Недиагональных элементов H_ij (Смешанные производные) ---
        for i in range(self.dim):
            # Начинаем со следующего элемента, чтобы не считать дважды (H_ij = H_ji)
            for j in range(i + 1, self.dim):
                # Вычисление f(x + delta*e_i + delta*e_j)
                x_plus_ij = x_vec.copy()
                x_plus_ij[i] += delta
                x_plus_ij[j] += delta
                f_plus_ij = self.func(x_plus_ij)

                # Формула: (f(x+di+dj) - f(x+di) - f(x+dj) + f(x)) / d^2
                Hij = (f_plus_ij - f_plus_i[i] - f_plus_i[j] + fx) / d2

                H[i, j] = Hij
                H[j, i] = Hij  # Симметричность
        return H

    def find_newton_direction(self, H: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Решает СЛАУ H * d = -g для нахождения Ньютоновского направления d.
        """
        # 1. Проверяем, что матрица H не вырождена (обратима)
        # Если определитель близок к нулю, матрица может быть сингулярной.
        if np.linalg.det(H) == 0:
            # В реальных методах оптимизации здесь часто переходят к методу
            # наискорейшего спуска или используют квази-ньютоновские методы.
            # В простейшем случае:
            print("Warning: Hessian matrix is singular. Falling back to Steepest Descent.")
            return -g  # Возвращаем антиградиент (направление Быстрого Спуска)

        # 2. Правая часть уравнения (b) равна -g
        b = -g

        # 3. Решение СЛАУ H * d = b
        # np.linalg.solve(A, b) решает A*x = b
        d = np.linalg.solve(H, b)

        return d

    def nelder_mead(self, x0: tuple, tol: float, delta: float = 1):
        x_vec = np.array(x0)
        result = [x_vec]
        result_func = [self.get_val(x_vec)]
        x_figure: list[np.array] = [np.copy(x_vec)]
        fx_figure: list[float] = [self.get_val(x_vec)]
        k = 0
        for d in range(self.dim):
            k += 1
            x_0 = np.copy(x_vec)
            x_0[d] = x_vec[d] + delta
            x_figure.append(x_0)
            fx_figure.append(self.get_val(x_0))
        while k < self.n_max:
            k += 1
            fw = max(fx_figure)
            h = fx_figure.index(fw)
            w = np.copy(x_figure[h])
            fx = min(fx_figure)
            l = fx_figure.index(fx)
            x = np.copy(x_figure[l if l != h else l - 1])
            result.append(x)
            result_func.append(fx)

            all_delta: list[float] = []
            for apex in x_figure:
                for d in range(self.dim):
                    all_delta.append(abs(x[d] - apex[d]))
            delta = max(all_delta)
            if delta <= tol:
                return result, result_func
            sum_vectors = np.sum(x_figure, axis=0)
            sum_vectors = sum_vectors - x_figure[h]
            c_centroid = sum_vectors / self.dim
            # x_figure.pop(h)
            # fx_figure.pop(h)
            # c_centroid = np.sum(x_figure, axis=0) / self.dim
            y = 2 * c_centroid - w
            fy = self.get_val(y)
            if fy < fx:
                z = 2 * y - c_centroid
                fz = self.get_val(z)
                if fz < fy:
                    # x_figure.append(z)
                    # fx_figure.append(fz)
                    x_figure[h] = z
                    fx_figure[h] = fz
                else:
                    # x_figure.append(y)
                    # fx_figure.append(fy)
                    x_figure[h] = y
                    fx_figure[h] = fy
                continue
            is_replaced = False
            for d in range(self.dim + 1):
                if d != h and fy < fx_figure[d]:
                    fx_figure[h] = fy
                    x_figure[h] = y
                    is_replaced = True
                    break
            if is_replaced:
                continue
            if fy < fw:
                w = y
                x_figure[h] = y
                fw = fy
                fx_figure[h] = fy
                z = 0.5 * (w + c_centroid)
                fz = self.get_val(z)
                if fz < fw :
                    x_figure[h] = z
                    fx_figure[h] = fz
                    continue
            for d in range(self.dim + 1):
                if d != l :
                    x_figure[d] = 0.5*(x_figure[d] + x)
                    fx_figure[d] = self.get_val(x_figure[d])
            continue



if __name__ == "__main__":
    # f = Koshi("0.1 * (12 + x_1**2 + (1 + x_2**2) / x_1**2 + ((x_1 * x_2)**2 + 100) / (x_1 * x_2)**4)")
    # f = Koshi()
    # f = Koshi(" (x_1 - 4)**2 + (x_2 - 4)**2 ")
    # f = Koshi(" 100 * (x_2 - x_1**2)**2 + (1 - x_1)**2 ")
    # f = Koshi(" (0.01 * (x_1 - 3))**2 - (x_2 - x_1) + exp(20 * (x_2 - x_1)) ")
    f = Koshi(" 100 * (x_2 - x_1**3 + x_1)**2 + (x_1 - 1)**2 ")

    res = f.quick_down((5, 20), 0.0000001)
    # res = f.fl_rivs((1.5, 3), 0.0000001)
    # res = f.polocoribyer((5, 70), 0.0000001)
    # res = f.newthon((5, 70), 0.0000001)
    # res = f.nelder_mead((5, 70), 0.0000001)

    # res = f.quick_down((-2.8, 8), 0.0000001)
    # res = f.polocoribyer((-2.8, 8), 0.0001)
    # res = f.fl_rivs((-2.8, 8), 0.000001)
    # res = f.newthon((-2.8, 8), 0.0000001)

    fig = None  # Инициализируем переменную для графика

    try:
        print(res[0])
        print(res[1])

        # 1. Создаем объект графика
        fig = plot_3d_surface_with_path(f.func, res[0], res[1], True, )
        plt.show(block=True)

    except Exception as e:
        print(f"ERROR USING pyplot: {e}")

    finally:
        # Этот код ГАРАНТИРОВАННО выполнится, даже при прерывании!
        if fig is not None:
            plt.close(fig)
    plt.close('all')
