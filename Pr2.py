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

def f4():
    print('Четвертое задание')
    
    Q = [1, 2, 3, 4, 2, 1, 3, 4, 5, 6, 5, 4, 3, 2]
    T = ['a', 'b', 'c', 'c', 'c', 'b', 'a', 'c', 'a', 'a', 'b', 'c', 'b', 'a']

    list={}
    
    for i in range(len(T)):
        x = list.get(T[i])
        if x == None:
            x = 0
        x = x + Q[i]
        list.update({T[i]:x})
    print(list)
f4()    


def f5():
    print('Пятое задание')
    
    data = fetch_california_housing(as_frame=True)
    u = pd.concat([data.frame, data.target], axis=1)
    print(u)
    print('\n')

    print('Использование метода info(): ')
    u.info()
    
    print('\n')
    
    print('Использование метода isna().sum(): ')
    print(u.isna().sum())
    
    print('\n')
    
    print('Вывод записи, где средний возраст домов в районе более 50 лет и население района более 2500 человек\n(через метод loc()): ')
    print(u.loc[(u['HouseAge'] > 50) & (u['Population'] > 2500)])
    print('\n')
    
    print ('Узнать max и min значения медиайнной стомости домов (max(), min()):')
    
    print('Для max():', u['MedHouseVal'].max())
    print('\n')
    print('Для min():', u['MedHouseVal'].min())
    print('\n')
    
    print('Вывести название признака и его среднее значение (apply()): ')
    print('\n')
    def k(x):
        mean = x.mean()
        return mean
    print(u.apply(k))
f5()