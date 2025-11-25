from typing import Callable, Any


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
