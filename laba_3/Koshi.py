import sympy as sp
import numpy as np


class Koshi:
    def __init__(self, func_str: str = "x_1**2 + (x_2 - 2)**2", dim: int = 2, n_max: int = 1000):
        self.func_str = func_str
        self.dim = dim
        self.func = self._create_func_wrapper()
        self.n_max = n_max

    def _create_func_wrapper(self):
        symbols_list = [sp.symbols(f'x_{i+1}') for i in range(self.dim)]
        f_sym = sp.sympify(self.func_str)
        f_lambda_base = sp.lambdify(symbols_list, f_sym, 'numpy')

        def func_wrapper(x_vec: np.ndarray) -> float:
            return f_lambda_base(*x_vec)

        return func_wrapper

    def get_val(self, x_vect: np.ndarray) -> float:
        return self.func(x_vect)



    def calculate_gradient(self, x_vec: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        g_vec = np.zeros(self.dim)
        fx_val = self.func(x_vec)
        y_vec = np.copy(x_vec)
        for i in range(self.dim):
            y_vec[i] = y_vec[i] + delta
            fy_val = self.func(y_vec)
            g_vec[i] = (fy_val - fx_val)/delta
            y_vec[i] = x_vec[i]
        return g_vec

    def corop_line(self, x: float, h: float, tol: float) -> tuple[float, float]:
        r = 0
        fx = self.get_val(x)
        while True and self.n_final < self.n_max:
            y = x + h
            fy = self.get_val(y)
            if fy >= fx:
                if r == 2:
                    r = 0.25
                else:
                    r = - 0.5
            else:
                x = y
                fx = fy
                if r >= 0.5:
                    r = 2
                else:
                    r = 0.5
            h = h * r
            print(f'uncertainty interval = {h}')
            if not abs(h) > tol:
                y = x + h
                fy = self.get_val(y)
                print(" - final check")
                if fy < fx:
                    return y, fy
                return x, fx

