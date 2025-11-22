from laba_1.graph_3d import build_3d_graph_swen
from laba_1.math_part import laba_1


def lab_1():
    func = input("input the function or press enter to use default values: ")
    if func:
        x0 = float(input("input x0 value: "))
        h = float(input("input h0 value: "))
        laba = laba_1(func, x0, h)
    else:
        laba = laba_1()
    n = input("input max number of iterations or press enter to use default value (20):  ")
    print("\nLaboratory job # 1 in progress:\n")
    if n:
        laba.swen.swen_method(int(n))
    else:
        laba.swen.swen_method()
    laba.final_print()
    laba.show_graph()


if __name__ == '__main__':
    laba_num = input('input number of LW (1, 2 is available): ')

    match laba_num:
        case '1':
            lab_1()
        case '2':
            print('in progress')
            print("Dichotomy method - 1")
