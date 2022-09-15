import numpy as np
import pandas as pd
import random
from sklearn.datasets import fetch_california_housing

def f1():
    print ('Первое задание')
    print('Программа вычисляет сумму квадратов всех записанных чисел, пока их сумма не будет равна нулю')
    print('Введите числа: ')
    first = int(input())
    second = int(input())
    quadro = first ** 2 + second ** 2
    sum = first + second
    while sum != 0:
        temp = int(input())
        sum += temp
        print ('Текущая сумма: ' + str(sum))
        quadro += temp ** 2
        if sum == 0:
            print ('Сумма квадратов всех чисел: ', quadro)
            print ('Сумма всех чисел (для проверки): ', sum)
            break
f1()

def f2():
    print('Второе задание')
    print('Введите число N: ') 
    n = int(input())
    l = []
    Up = 0
    Upcheck = 0
    while Up != n:
        Up += 1
        while Upcheck != Up:
            Upcheck += 1
            l.append(Up)
        Upcheck = 0
    print('Результат работы програмы: ', l)
f2()

def f3():
    print('Третье задание')
    i = random.randint(1, 10)
    g = random.randint(1, 10)
    print('Число i: ', i)
    print('Число g: ', g)
    a = np.random.randint(0, 15, size = (i, g))
    print('Вывод оригинальной матрицы: \n', a)
    print('Разложенный массив через reshape (по горизонтали): \n', a.reshape(1, i * g))
    print('Разложенная матрица через reshape (по вертикали)', a.reshape(i * g, 1))
    print('Разложенная матрица через flatten (F): \n', a.flatten('F'))
f3()