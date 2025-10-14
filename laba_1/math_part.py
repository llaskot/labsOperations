import sympy as sp


class laba_1:
    def __init__(self, func_str="x**2+2*x", x_0=5.0, h0=1.0):
        self.fc = None
        self.fb = None
        self.fa = None
        self.n = None
        self.c = None
        self.b = None
        self.a = None
        self.h_final = None
        self.x_arr = None
        self.fx_arr = None
        self.func_str = func_str
        self.func = sp.lambdify(sp.symbols('x'), sp.sympify(func_str))
        self.x0 = x_0
        self.h = h0

    def find_val(self, x):
        return self.func(x)

    def swen_method(self, max_steps: int = 20):
        self.x_arr = []
        self.fx_arr = []
        c = self.x0
        h = self.h
        fc = self.find_val(c)
        print(f'k=0, {h=}, x0 = {c}, fx0 = {fc}')

        b = c + h
        fb = self.find_val(b)
        print(f'k=1, {h=}, x1 = {b}, fx1 = {fb}')

        self.x_arr.append(c)
        self.x_arr.append(b)
        self.fx_arr.append(fc)
        self.fx_arr.append(fb)
        if fb >= fc:
            a = c - h
            fa = self.find_val(a)
            self.x_arr.append(a)
            self.fx_arr.append(fa)
            print(f'k=2, {h=}, x2 = {a}, fx2 = {fa}')
            if fa >= fc:
                self.a, self.b, self.c, self.n = a, b, c, 2
                self.fa, self.fb, self.fc, self.h_final = fa, fb, fc, h
            else:
                self.swen_inc(c, fc, a, fa, 2 * h, max_steps)
        else:
            self.swen_dec(c, fc, b, fb, h * 2, max_steps)
        print(f'uncertainty interval (a:b) = {self.a} : {self.b},\nf(a) = {self.fa},\nf(b) = {self.fb},\n'
              f'c = {self.c},\nf(c) = {self.fc},\nfunction-using number = {self.n}')
        print("k:   ", "  ".join(f"{i:>5}" for i in range(len(self.x_arr))))
        print("x:   ", "  ".join(f"{v:>5}" for v in self.x_arr))
        print("f(x):", "  ".join(f"{v:>5}" for v in self.fx_arr))

    def swen_dec(self, a, fa, c, fc, h, max_steps):
        n = 2
        while n < max_steps:
            b = c + h
            fb = self.find_val(b)
            self.x_arr.append(b)
            self.fx_arr.append(fb)
            print(f'k={n}, {h=}, x{n} = {b}, fx{n} = {fb}')
            if fb < fc:
                a, fa, c, fc, h = c, fc, b, fb, 2 * h
                n += 1
            else:
                self.a, self.b, self.c, self.n = a, b, c, n
                self.fa, self.fb, self.fc, self.h_final = fa, fb, fc, h
                return

    def swen_inc(self, b, fb, c, fc, h, max_steps):
        n = 3
        while n < max_steps:
            a = c - h
            fa = self.find_val(a)
            self.x_arr.append(a)
            self.fx_arr.append(fa)
            print(f'k={n}, {h=}, x{n} = {a}, fx{n} = {fa}')
            if fa >= fc:
                self.a, self.b, self.c, self.n, self.h_final = a, b, c, n, h
                self.fa, self.fb, self.fc = fa, fb, fc
                return
            else:
                b, fb, c, fc, h = c, fc, a, fa, 2 * h
                n += 1


if __name__ == "__main__":
    laba = None
    func = input("input the function or press enter to use default values: ")
    if func:
        x0 = float(input("input x0 value: "))
        h = float(input("input x0 value: "))
        laba = laba_1(func, x0, h)
    else:
        laba = laba_1()
    n = int(input("input max number of iterations: "))
    laba.swen_method(n)
