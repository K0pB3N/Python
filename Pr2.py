import math

def f1():

    first = int(input())
    second = int(input())
    quadro = first ** 2 + second ** 2
    sum = first + second
    while sum != 0:
        temp = int(input())
        sum += temp
        quadro += temp ** 2
        if sum == 0:
            print ('Сумма квадратов всех чисел: ', quadro)
            break
f1()