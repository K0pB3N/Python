import math

def f1():
    
    print('Даны стороны треугольник: 7, 9, 10\n')
    a = 7
    b = 9
    c = 10
    p = (a + b + c) / 2
    res_of_triangle = math.sqrt(p * (p - a) * (p - b) * (p - c))
    print('Площадь треугольника по формуле Герона: ', res_of_triangle , '\n')
    print('Дан радиус круга: 5\n')
    r = 5
    res_of_circle = math.pi * r ** 2
    print('Площадь круга по формуле: ', res_of_circle , '\n')
    print('Даны стороны прямоугольника: 7, 13\n')
    g = 7
    q = 13
    res_of_rectangl = g * q
    print('Площадь прямоугольника по формуле: ', res_of_rectangl , '\n')
    dict = {'Треугольник': res_of_triangle,
            'Круг': res_of_circle, 'Прямоугольник': res_of_rectangl}
    print(dict, type(dict))
f1()

print ('\n')

def f2():
        
        play = ['+', '-', '*', '/', '**', '//']
        while play != None:
                print('Введите два числа или введите СТОП, чтобы закончить: \n')
                a = int(input())
                b = int(input())
                if a == 'СТОП' or b == 'СТОП':
                        break
                print('Выбирите действие или введите СТОП, чтобы закончить: ', play, '\n')
                x = input()
                print('\n')
                print('Ваши числа: ', a, ',', b, '\n')
                if x == '+':
                    play.remove(x)
                    print('Результат операции сложения:', a + b, '\n')
                elif x == '-':
                    play.remove(x)
                    print('Результат операции вычитания: ', a - b, '\n')
                elif x == '*':
                    play.remove(x)
                    print('Результат операции умножения: ', a * b, '\n')
                elif x == '/':
                    play.remove(x)
                    print('Результат операции деления: ', a / b, '\n')
                elif x == '**':
                    play.remove(x)
                    print('Результат операции возведения в степень: ', a ** b, '\n')
                elif x == '//':
                    play.remove(x)
                    print('Результат операции целочисленного деления: ', a // b, '\n')
                elif x == 'СТОП':
                    print('Конец программы')
                    break           
f2()

    
