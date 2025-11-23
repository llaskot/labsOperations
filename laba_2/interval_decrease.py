from math import sqrt

import sympy as sp

from laba_1.graph_3d import build_3d_decreasing
from laba_1.swen_class import Swen


class IntervalDecrease:
    def __init__(self, func_str="x**2+2*x", x_0=5.0, h0=1.0):
        self.func_str = func_str
        self.func = sp.lambdify(sp.symbols('x'), sp.sympify(func_str))
        self.x0 = x_0
        self.h = h0
        self.swen = Swen(self.func, self.x0, self.h)
        self.n_final = -1
        self.all_x = []
        self.all_fx = []
        self.a = 0
        self.b = 0

    def find_a_b(self):
        self.swen.swen_method(50)
        self.a = self.swen.a
        self.b = self.swen.b

    @staticmethod
    def iteration_print(func):
        def inner(self, x, *args, **kwargs):
            y = func(self, x, *args, **kwargs)
            self.n_final += 1
            print(
                f"point # {self.n_final}, "
                # f"cur step = {self.h_final}, "
                f"x{self.n_final} = {x}, "
                f"f(x{self.n_final}) = {y}"
            )
            self.all_x.append(x)
            self.all_fx.append(y)
            return y

        return inner

    @iteration_print
    def get_val(self, x):
        return self.func(x)

    def dichotomy(self, a: float, b: float, shift: float, tol: float) -> tuple[float, float]:
        self.n_final = -1
        self.all_x = []
        self.all_fx = []
        while True:
            c = (a + b) / 2
            u = c - shift
            v = c + shift
            fu = self.get_val(u)
            fv = self.get_val(v)
            if fu < fv:
                b = v
            else:
                a = u

            if not b - a > tol:
                return a, b


    def binary(self, a, b, tol):
        self.n_final = -1
        self.all_x = []
        self.all_fx = []
        c = (a + b) / 2
        while True:
            u = (a + c) / 2
            fc = self.get_val(c)
            fu = self.get_val(u)
            if fu < fc:
                b = c
                c = u
                fc = fu
            else:
                v = (c + b) / 2
                fv = self.get_val(v)
                if fv < fc:
                    a = u
                    c = v
                    fc = fv
                else:
                    a = u
                    b = v
            if (c - a) <= tol:
                return

    def gold(self, a, b, tol):
        self.n_final = -1
        self.all_x = []
        self.all_fx = []

        t = 0.5 * (sqrt(5) - 1)
        h = t * (b - a)
        x = b - h
        fx = self.get_val(x)
        y = a + h
        fy = self.get_val(y)
        while True:
            if not fx > fy:
                b = y
                y = x
                fy = fx
                d = b - a
                x = b - t * d
                fx = self.get_val(x)
            else:
                a = x
                x =y
                fx = fy
                d = b-a
                y = a+t*d
                fy = self.get_val(y)
            if not d > tol:
                if fy > fx:
                    x = y
                    fx = fy
                return

    def corop(self, x, h, tol):
        self.n_final = -1
        self.all_x = []
        self.all_fx = []
        r = 0
        fx = self.get_val(x)
        while True:
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
            h = h*r
            if not abs(h) > tol:
                return

    def show(self):
        build_3d_decreasing(self.all_x, self.all_fx, self.func, self.func_str)


if __name__ == "__main__":
    cl = IntervalDecrease()
    cl.find_a_b()
    print(f'{cl.a=} , {cl.b=}')
    res = cl.dichotomy(cl.a, cl.b, 0.001, 0.01)
    # cl.binary(cl.a, cl.b, 0.05)
    # cl.gold(cl.a, cl.b, 0.05)
    # cl.corop(cl.x0, cl.h, 0.05)

    print(cl.all_x)
    print(cl.all_fx)
    # print(f'{cl.a=}, {cl.b=}')
    print(res)
    cl.show()
