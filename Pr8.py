import pandas as pd
import numpy as np
import scipy.stats as stast
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols 
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_csv('Pr8insurance.csv')

def f1():
    print(data, '\n')
    print('Уникальные регионы: \n',pd.unique(data.region), '\n')

def f2():
    groups = data.groupby('region').groups # группировка по регионам
    
    southwest = data.bmi[groups['southwest']] # bmi для южного региона
    southeast = data.bmi[groups['southeast']] # bmi для юго-восточного региона
    northwest = data.bmi[groups['northwest']] # bmi для северо-западного региона
    northeast = data.bmi[groups['northeast']] # bmi для северо-восточного региона
    
    print(stast.f_oneway(southwest, southeast, northwest, northeast)) # проверка гипотезы о равенстве средних bmi для всех регионов
    print('\n')

def f3():
    model = ols('bmi ~ region', data = data).fit() # модель
    anova_results = sm.stats.anova_lm(model, typ=2) # результаты анализа дисперсии
    print(anova_results)
    print('\n')

def f4():
    region = ['southwest', 'southeast', 'northwest', 'northeast']
    region_pairs = [] 
    
    for i in range (3): # создание списка пар регионов
        for j in range (i+1, 4):
            region_pairs.append([region[i], region[j]])
    
    for i, j in region_pairs: # вывод результатов теста Стьюдента для каждой пары регионов
        print(i, j, '\n', stast.ttest_ind(data.bmi[data.region == i], data.bmi[data.region == j]))
    
    x = 0.05 / 6 # уровень значимости для каждой пары регионов
    print('\n', 'Поправка Бонферрони: ', round(x, 3), '\n')

def f5():
    tukey = pairwise_tukeyhsd(endog=data.bmi, groups=data.region, alpha=0.05)  # тест Тьюки
    tukey.plot_simultaneous() # график
    plt.vlines(x = 31.25, ymin = -0.5, ymax = 3.5, colors = 'r') # вертикальная линия
    print(tukey.summary()) # вывод результатов теста Тьюки
    plt.show()
    print('\n')



def f6():
    model = ols('bmi ~ C(region) + C(sex) + C(region):C(sex)', data=data).fit() # модель
    print(sm.stats.anova_lm(model, typ=2)) # результаты анализа дисперсии
    print('\n')



def f7():
    data['combination'] = data.sex + ' / ' + data.region # создание нового столбца
    tukey = pairwise_tukeyhsd(endog=data.bmi, groups=data.combination, alpha=0.05) # тест Тьюки
    tukey.plot_simultaneous() # график
    plt.vlines(x = 31.5, ymin = -0.5, ymax = 7.5, colors = 'r') # вертикальная линия
    print(tukey.summary()) # вывод
    plt.show()
    print('\n')


if __name__ == '__main__':
    msg_dic = {
        1: 'Задание 1',
        2: 'Задание 2',
        3: 'Задание 3',
        4: 'Задание 4',
        5: 'Задание 5',
        6: 'Задание 6',
        7: 'Задание 7',
        0: 'Выход'
    }
    digit = [1, 2, 3, 4, 5, 6, 7]
    msg = 'Выберите задание:', digit
    while digit != None:
        print(msg)
        choice = input('Выберите задание: ').strip()
        if choice.isdigit():
            choice = int(choice)
            if choice in msg_dic.keys():
                print(msg_dic[choice])
                if choice == 1:
                    f1()
                    digit.remove(choice)
                elif choice == 2:
                    f2()
                    digit.remove(choice)
                elif choice == 3:
                    f3()
                    digit.remove(choice)
                elif choice == 4:
                    f4()
                    digit.remove(choice)
                elif choice == 5:
                    f5()
                    digit.remove(choice)
                elif choice == 6:
                    f6()
                    digit.remove(choice)
                elif choice == 7:
                    f7()
                    digit.remove(choice)
                else:
                    print('Ошибка')
            else:
                print('Ошибка')
        if digit == []:
            digit = None
            print('\n', 'Все задания выполнены')
            break
        if choice == 0:
            digit = None
            break
