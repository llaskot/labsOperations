from math import sqrt
import sympy as sp
from laba_1.graph_3d import build_3d_decreasing
from laba_1.swen_class import Swen


class IntervalDecrease:
    def __init__(self, func_str: str = "x**2+2*x", x_0: float = 5.0, h0: float = 1.0, n_max: int = 50):
        self.func_str = func_str
        self.func = sp.lambdify(sp.symbols('x'), sp.sympify(func_str))
        self.x0 = x_0
        self.h = h0
        self.swen = Swen(self.func, self.x0, self.h)
        self.n_max = n_max
        self.n_final = -1
        self.all_x = []
        self.all_fx = []
        self._a = None
        self._b = None
        self.x_final = None
        self.fx_final = None

    def _find_a_b(self):
        self.swen.swen_method(self.n_max)
        self._a = self.swen.a
        self._b = self.swen.b

    @property
    def a(self):
        if self._a is None:
            self._find_a_b()
        return self._a

    @property
    def b(self):
        if self._b is None:
            self._find_a_b()
        return self._b

    @staticmethod
    def clean(func):
        def inner(self, *args, **kwargs):
            self.n_final = -1
            self.all_x = []
            self.all_fx = []
            result = func(self, *args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                self.x_final = result[0]
                self.fx_final = result[1]
            else:
                self.x_final = None
                self.fx_final = None
            return result
        return inner

    @staticmethod
    def iteration_print(func):
        def inner(self, x, *args, **kwargs):
            y = func(self, x, *args, **kwargs)
            self.n_final += 1
            print(
                f"point # {self.n_final}, "
                f"x{self.n_final} = {x}, "
                f"f(x{self.n_final}) = {y}", end=", "
            )
            self.all_x.append(x)
            self.all_fx.append(y)
            return y

        return inner

    @iteration_print
    def get_val(self, x):
        return self.func(x)

    @clean
    def dichotomy(self, a: float, b: float, shift: float, tol: float) -> tuple[float, float]:
        while True and self.n_final < self.n_max:
            c = (a + b) / 2
            u = c - shift
            v = c + shift
            fu = self.get_val(u)
            fv = self.get_val(v)
            if fu < fv:
                b = v
            else:
                a = u
            print(f'uncertainty interval = {b - a}')
            if not b - a > tol:
                x_final = (a + b) / 2
                y_final = self.get_val(x_final)
                print(" - final values")
                return x_final, y_final

    @clean
    def binary(self, a: float, b: float, tol: float) -> tuple[float, float]:
        c = (a + b) / 2
        fc = self.get_val(c)
        while True and self.n_final < self.n_max:
            u = (a + c) / 2
            fu = self.get_val(u)
            if fu < fc:
                b = c
                c = u
                fc = fu
            else:
                v = (c + b) / 2
                fv = self.get_val(v)
                if fv < fc:
                    a = c
                    c = v
                    fc = fv
                else:
                    a = u
                    b = v
            print(f'uncertainty interval = {b - a}')
            if (b - a) <= tol:
                x_final = (a + b) / 2
                y_final = self.get_val(x_final)
                print(" - final values")
                return x_final, y_final

    @clean
    def gold(self, a: float, b: float, tol: float) -> tuple[float, float]:
        t = 0.5 * (sqrt(5) - 1)
        h = t * (b - a)
        x = b - h
        fx = self.get_val(x)
        y = a + h
        fy = self.get_val(y)
        while True and self.n_final < self.n_max:
            if fx > fy:
                a = x
                x = y
                fx = fy
                d = b - a
                y = a + t * d
                fy = self.get_val(y)
            else:
                b = y
                y = x
                fy = fx
                d = b - a
                x = b - t * d
                fx = self.get_val(x)
            print(f'uncertainty interval = {d}')
            if not d > tol:
                if fy < fx:
                    x = y
                    fx = fy
                return x, fx

    @clean
    def corop(self, x: float, h: float, tol: float) -> tuple[float, float]:
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

    def show_graph(self):
        build_3d_decreasing(self.all_x, self.all_fx, self.func, self.func_str)

    def find_analytical_extremum(self):

        x = sp.symbols('x')
        f_sym = sp.sympify(self.func_str)
        f_prime_sym = sp.diff(f_sym, x, 1)
        f_double_prime_sym = sp.diff(f_sym, x, 2)

        # print(f"\nAnalytical f'(x) = {f_prime_sym}")
        # print(f"Analytical f''(x) = {f_double_prime_sym}")

        try:
            solutions = sp.solve(f_prime_sym, x)
        except NotImplementedError:
            print("❌ Impossible f'(x) = 0 in Analytical way.")
            return None, f_prime_sym, f_double_prime_sym

        # print(f"\nextremes: {solutions}")

        return solutions[0], f_prime_sym, f_double_prime_sym

    def final_print(self):
        x, f1, f2 = self.find_analytical_extremum()
        f1 = sp.lambdify(sp.symbols('x'), sp.sympify(f1))
        f2 = sp.lambdify(sp.symbols('x'), sp.sympify(f2))
        print(
            f'\nFinal print:\n'
            f'function-using number = {self.n_final + 1}\n'

            f'The best x = {self.x_final} appropriate f(x) = {self.fx_final},\n'
            
            f"f'({self.x_final}) = {f1(self.x_final)},\n"
            f"f''({self.x_final}) = {f2(self.x_final)},\n"
        )



# if __name__ == "__main__":
#     cl = IntervalDecrease()
#     # cl.find_a_b()
#     # print(f'{cl.a=} , {cl.b=}')
#     # res = cl.dichotomy(cl.a, cl.b, 0.0001, 0.001)
#     # res = cl.binary(cl.a, cl.b, 0.001)
#     res = cl.gold(cl.a, cl.b, 0.001)
#     # res = cl.corop(cl.x0, cl.h, 0.001)
#
#     print(cl.all_x)
#     print(cl.all_fx)
#     print(f'{cl.a=}, {cl.b=}')
#     print(res)
#     cl.show_graph()
#     x_extr, derivative_1, derivative_2 = cl.find_analytical_extremum()
#     print(f"\nExtremum: x* = {x_extr}, f(x*) = {cl.func(x_extr)}")
#
#     # res = cl.gold(cl.a, cl.b, 0.001)
#     res = cl.corop(cl.x0, cl.h, 0.001)
#
#     print(cl.all_x)
#     print(cl.all_fx)
#     print(f'{cl.a=}, {cl.b=}')
#     print(res)
#     cl.show_graph()
#     # x_extr, derivative_1, derivative_2 = cl.find_analytical_extremum()
#     # print(f"\nExtremum: x* = {x_extr}, f(x*) = {cl.func(x_extr)}")
