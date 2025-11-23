
import sympy as sp

from laba_1.graph_3d import build_3d_graph_swen
from laba_1.swen_class import Swen


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


