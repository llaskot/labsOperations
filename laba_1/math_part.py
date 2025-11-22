from typing import Callable, Any

import sympy as sp

from laba_1.graph_3d import build_3d_graph_swen


class laba_1:
    def __init__(self, func_str="x**2+2*x", x_0=5.0, h0=1.0):
        self.func_str = func_str
        self.func = sp.lambdify(sp.symbols('x'), sp.sympify(func_str))
        self.x0 = x_0
        self.h = h0
        self.swen = Swen(self.func, self.x0, self.h)

    def final_print(self):
        print(
            f'\nFinal print:\n'
            f'function-using number = {self.swen.n_final + 1}\n'
            f'last step = {self.swen.h_final}\n'

            f'uncertainty interval (a:b) = {self.swen.a} : {self.swen.b},\n'
            f'f(a) = {self.swen.fa},\nf(b) = {self.swen.fb},\n'
            f'c = {self.swen.c},\nf(c) = {self.swen.fc},\n'
        )
        print("k:   ", "  ".join(f"{i:>5}" for i in range(len(self.swen.all_x))))
        print("x:   ", "  ".join(f"{v:>5}" for v in self.swen.all_x))
        print("f(x):", "  ".join(f"{v:>5}" for v in self.swen.all_fx))

    def show_graph(self):
        build_3d_graph_swen(self.swen.all_x,
                            self.swen.all_fx,
                            (self.swen.a, self.swen.fa),
                            (self.swen.b, self.swen.fb),
                            (self.swen.c, self.swen.fc),
                            self.func,
                            self.func_str)


class Swen:
    def __init__(self, my_func: Callable[..., Any], x_0: float, h_0: float):
        """
        income
        :param my_func: function to process
        :param x_0: start point
        :param h_0: start step
        :param max_num: max number of steps to prevent endless
        """
        self.my_func = my_func
        self.x_0 = x_0
        self.h_0 = h_0
        # self.max_num = max_num

        """
        outcome
        self.all_x: all x vals
        self.all_fx: all f(x) vals
        self.a: start uncertainty interval
        self.b: end uncertainty interval
        self.fa: f(start uncertainty interval)
        self.fb: f(end uncertainty interval)
        self.c: inner point of uncertainty interval
        self.fc: f(inner point of uncertainty interval)
        self.n_final: real number of iterations
        self.h_final: last step value
        """
        self.all_x: list[float] = []
        self.all_fx: list[float] = []
        self.a: float = 0.0
        self.b: float = 0.0
        self.fa: float = 0.0
        self.fb: float = 0.0
        self.c: float = 0.0
        self.fc: float = 0.0
        self.n_final: int = -1
        self.h_final: float = 0.0

    @staticmethod
    def iter_counter_deck(func):
        def inner(self, *args, **kwargs):
            self.n_final += 1
            return func(self, *args, **kwargs)

        return inner

    # def _iteration_print(self, x, y):
    #     print(f'point # {self.n_final}, cur step = {self.h_final}, x{self.n_final} = {x},  f(x{self.n_final}) = {y}')

    @staticmethod
    def iteration_print(func):
        def inner(self, x, *args, **kwargs):
            y = func(self, x, *args, **kwargs)
            print(
                f"point # {self.n_final}, "
                f"cur step = {self.h_final}, "
                f"x{self.n_final} = {x}, "
                f"f(x{self.n_final}) = {y}"
            )
            return y

        return inner

    @iteration_print
    @iter_counter_deck
    def find_val(self, x):
        return self.my_func(x)

    def swen_method(self, max_steps: int = 20):
        c = self.x_0
        self.h_final = self.h_0
        fc = self.find_val(c)
        # self._iteration_print(c, fc)
        b = c + self.h_final
        fb = self.find_val(b)
        # self._iteration_print(b, fb)
        self.all_x.append(c)
        self.all_x.append(b)
        self.all_fx.append(fc)
        self.all_fx.append(fb)
        if fb >= fc:
            a = c - self.h_final
            fa = self.find_val(a)
            self.all_x.append(a)
            self.all_fx.append(fa)
            # self._iteration_print(a, fa)
            if fa >= fc:
                self.a, self.b, self.c = a, b, c
                self.fa, self.fb, self.fc = fa, fb, fc
            else:
                self._swen_inc(c, fc, a, fa, max_steps)
        else:
            self._swen_dec(c, fc, b, fb, max_steps)

    def _swen_dec(self, a, fa, c, fc, max_steps):
        while self.n_final < max_steps:
            self.h_final *= 2
            b = c + self.h_final
            fb = self.find_val(b)
            self.all_x.append(b)
            self.all_fx.append(fb)
            # self._iteration_print(b, fb)

            if fb < fc:
                a, fa, c, fc = c, fc, b, fb
            else:
                self.a, self.b, self.c = a, b, c
                self.fa, self.fb, self.fc = fa, fb, fc
                return

    def _swen_inc(self, b, fb, c, fc, max_steps):
        while self.n_final < max_steps:
            self.h_final *= 2
            a = c - self.h_final
            fa = self.find_val(a)
            self.all_x.append(a)
            self.all_fx.append(fa)
            # self._iteration_print(a, fa)
            if fa >= fc:
                self.a, self.b, self.c = a, b, c
                self.fa, self.fb, self.fc = fa, fb, fc
                return
            else:
                b, fb, c, fc = c, fc, a, fa


# if __name__ == "__main__":
#     # laba = laba_1(func_str="x**2+2*x", x_0=-10.0, h0=1.0)
#     laba = laba_1()
#
#     laba.swen.swen_method(6)
#     laba.final_print()
#     laba.show_graph()
