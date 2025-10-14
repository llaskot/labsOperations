
from laba_1.graph_3d import build_3d_graph_swen
from laba_1.math_part import laba_1





# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    laba = None
    func = input("input the function or press enter to use default values: ")
    if func:
        x0 = float(input("input x0 value: "))
        h = float(input("input x0 value: "))
        laba = laba_1(func, x0, h)
    else:
        laba = laba_1()
    n = input("input max number of iterations: ")
    if n:
        laba.swen_method(int(n))
    else:
        laba.swen_method()
    build_3d_graph_swen(laba.x_arr, laba.fx_arr, (laba.a, laba.fa), (laba.b, laba.fb),
                        (laba.c, laba.fc), laba.func, laba.func_str)
