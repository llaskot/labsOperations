import sympy as sp


class laba_1:
    def __init__(self, func_str="x**2+2*x", x_0=5, h=1):
        self.func = sp.lambdify(sp.symbols('x'), sp.sympify(func_str))
        self.x0 = x_0
        self.h = h

    def find_val(self, x):
        return self.func(x)


if __name__ == "__main__":
    l = laba_1()
    print(l.find_val(-5))
