from laba_1.math_part import laba_1
from laba_2.interval_decrease import IntervalDecrease


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
    return


def lab_2():
    func = input("input the function or press enter to use default values: ")
    if func:
        x0 = float(input("input x0 value: "))
        h = float(input("input h0 value: "))
        n = int(input("input max N value: "))
        laba = IntervalDecrease(func, x0, h, n)
    else:
        laba = IntervalDecrease()
    while True:
        print('''\nSelect method:
        1 - dichotomy method
        2 - binary method
        3 - golden ratio method
        4 - step adaptation method
        5 - exit''')
        method = input("Input number: ")
        match method:
            case '1':
                tol = float(input("Input tolerance: "))
                delta = float(input("Input probing distance (at least 3x less than tolerance): "))
                print('\nDichotomy method is used!\nUncertainty interval calculated by Swen`s method:\n '
                      f'a = {laba.a} : b = {laba.b}')
                laba.dichotomy(laba.a, laba.b, delta, tol)
                laba.final_print()
                laba.show_graph()

            case '2':
                tol = float(input("Input tolerance: "))
                print('\nBinary method is used!\nUncertainty interval calculated by Swen`s method:\n '
                      f'a = {laba.a} : b = {laba.b}')
                laba.binary(laba.a, laba.b, tol)
                laba.final_print()
                laba.show_graph()
            case '3':
                tol = float(input("Input tolerance: "))
                print('\nGolden ratio method is used!\nUncertainty interval calculated by Swen`s method:\n '
                      f'a = {laba.a} : b = {laba.b}')
                laba.gold(laba.a, laba.b, tol)
                laba.final_print()
                laba.show_graph()
            case '4':
                x0 = input(f"Input x0 or enter to use default ({laba.x0}): ")
                x0 = float(x0) if x0 else laba.x0
                h0 = input(f"fInput first step length or enter to use default ({laba.h}): ")
                h0 = float(h0) if h0 else laba.h
                tol = float(input("Input tolerance (required): "))
                print(f'\nStep adaptation method is used!\nFirst point = {x0}, First step = {h0}')
                laba.corop(x0, h0, tol)
                laba.final_print()
                laba.show_graph()
            case '5':
                return


def labs():
    while True:
        laba_num = input('\nInput number of LW (1, 2 is available) any else input for exit: ')

        match laba_num:
            case '1':
                lab_1()
            case '2':
                lab_2()
            case _:
                return


if __name__ == '__main__':
    labs()
