from this import d
import pandas as pd
import numpy as np
import scipy.stats as sts
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


def f1(): # Загрузить данные из файла ECDCCases.csv
    print('\n')
    print('1) Загрузить данные из файла ECDCCases.csv')
    data = pd.read_csv('ECDCCases.csv')
    print(data)
f1()
print('\n')

def f2(): # Проверить в данных наличие пропусков и вывести их количество
    print('2) Проверить в данных наличие пропусков и вывести их количество')
    data = pd.read_csv('ECDCCases.csv')
    print(data.isna().sum())
f2()

print('\n')

def f3(): # Вывести количество пропущенных значений в процентах
    print('3) Вывести количество пропущенных значений в процентах')
    data = pd.read_csv('ECDCCases.csv')
    for column in data.columns:
        missing = np.mean(data[column].isna()*100)
        print(f'{column}: {round(missing, 1)}%')
f3()

print('\n')

def f4(): #Удалить два признака из датасета, в которых больше всех пропусков. Для оставшихся обработать пропуски: для категориального использовать заполнение значением по умолчанию, для количественного – медианой.
    print('4) Удалить два признака из датасета, в которых больше всех пропусков. Для оставшихся обработать пропуски: для категориального использовать заполнение значением по умолчанию, для количественного – медианой.')
    data = pd.read_csv('ECDCCases.csv')
    data.drop(['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000', 'geoId'],
              axis=1,
              inplace=True)
    median_data = data.popData2019.median()
    data.popData2019.fillna(median_data, inplace=True)
    terr_code = 'other'
    data.countryterritoryCode.fillna(terr_code, inplace=True)
    print(data.isna().sum())
f4()
print('\n')

def f5(): # Посмотреть статистику по данным, используя метод describe. Сделать выводы о том, какие признаки содержат выбросы.
    print('5) Посмотреть статистику по данным, используя метод describe. Сделать выводы о том, какие признаки содержат выбросы.')
    data = pd.read_csv('ECDCCases.csv')
    print(data.describe())
f5()

print('\n')

def f6(): # Посмотреть, для каких стран количество смертей превысило 3000 за день и сколько таких дней.
    print('6) Посмотреть, для каких стран количество смертей превысило 3000 за день и сколько таких дней.')
    data = pd.read_csv('ECDCCases.csv')
    filter = data['deaths'] > 3000
    data_filter = data[['countriesAndTerritories', 'deaths', 'dateRep']].loc[filter]
    print(data_filter)
f6()

print('\n')

def f7(): # Вывести дубликаты в данных, если они есть.
    print('7) Вывести дубликаты в данных, если они есть.')
    data = pd.read_csv('ECDCCases.csv')
    due_rows = data[data.duplicated()]
    print(due_rows)
f7()

print('\n')

def f8(): # Удалить дубликаты в данных, если они есть.
    print('8) Удалить дубликаты в данных, если они есть.')
    data = pd.read_csv('ECDCCases.csv')
    data.drop_duplicates(inplace=True)
    print(data)
f8()

print('\n')

def f9(): # Загрузить данные из файла bmi.csv. Взять две выборки. Одна - это индекс массы тела для людей с региона northwest, вторая - для региона с southwest
    print('9) Загрузить данные из файла bmi.csv. Взять две выборки. Одна - это индекс массы тела для людей с региона northwest, вторая - для региона с southwest')
    data = pd.read_csv('bmi.csv')
    data1 = data.loc[data['region'] == 'northwest']
    print(data1)
    print('\n')
    data2 = data.loc[data['region'] == 'southwest']
    print(data2)
    print('\n')
    def f10(): # Сравнить средние значения этих выборок, используя t-тест Стьюдента. Предварительно проверить данные на нормальность и на гомогенность дисперсий.
        print('9.1) Сравнить средние значения этих выборок, используя t-тест Стьюдента. Предварительно проверить данные на нормальность и на гомогенность дисперсий.')
        res1 = sts.shapiro(data1['bmi'])
        res2 = sts.shapiro(data2['bmi'])
        print('\n')
        print(res1, '\n', res2)
        print('\n')
        print('9.2) Проверить данные на нормальность и на гомогенность дисперсий.')
        res_bar = sts.bartlett(res1, res2)
        print('\n')
        print(res_bar)
        print('\n')
        print('9.3) Сравнить средние значения этих выборок, используя t-тест Стьюдента.')
        res_test = sts.ttest_ind(res1, res2)
        print('\n')
        print(res_test)
        print('\n')
    f10() 
f9()

print('\n')

def f13():  # С помощью критерия Хи-квадрат проверить, является ли полученное распределение равномерным. Использовать функцию scipy.stats.chisquare().
    print('10) С помощью критерия Хи-квадрат проверить, является ли полученное распределение равномерным. Использовать функцию scipy.stats.chisquare().')
    d = {
        'Points': [1, 2, 3, 4, 5, 6],
        'Полученный': [97, 98, 109, 95, 97, 104],
        'Ожидаемый': [100, 100, 100, 100, 100, 100]
    }
    data = pd.DataFrame(data=d)
    print(sts.chisquare(data['Полученный'], data['Ожидаемый']))
f13()

print('\n')


def f14():  # С помощью критерия Хи-квадрат проверить, являются ли переменные зависимыми. Использовать функцию scipy.stats.chi2_contingency(). Влияет ли семейное положение на занятость?
    print('11) С помощью критерия Хи-квадрат проверить, являются ли переменные зависимыми. Использовать функцию scipy.stats.chi2_contingency(). Влияет ли семейное положение на занятость?')
    data = pd.DataFrame({
        'Женат': [89, 17, 11, 43, 22, 1],
        'Гражданский брак': [80, 22, 20, 35, 6, 4],
        'Не состоит в отношениях': [35, 44, 35, 6, 8, 22]
    })
    data.index = [
        'Полный рабочий день', 'Частичная занятость', 'Временно не работает',
        'На домохозяйстве', 'На пенсии', 'Учёба'
        ]
    print(sts.chi2_contingency(data)[1])
f14()