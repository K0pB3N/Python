from numpy.random import randint
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


def f1():
    print('Задание 1: Загрузить данные из файла “insurance.csv”', '\n' 
          'С помощью метода describe() посмотреть статистику по данным. Сделать выводы.', '\n')
    data = pd.read_csv('insurance.csv', sep=',')
    print(data.describe())

print('\n')

def f2():
    print('Задание 2: Построить гистограммы для числовых показателей', '\n')
    data = pd.read_csv('insurance.csv', sep=',')
    data.hist(color='green', edgecolor='black')
    plt.show()


def f3_4_5_6():
    data = pd.read_csv('insurance.csv', sep=',')
    
    mean_bmi = np.mean(data['bmi'])
    moda_bmi = sts.mode(data['bmi'])
    med_bmi = np.median(data['bmi'])
    std_bmi = data['bmi'].std()
    raz_bmi = data['bmi'].max() - data['bmi'].min()
    q1_bmi = np.percentile(data['bmi'], 25, interpolation='midpoint')
    q3_bmi = np.percentile(data['bmi'], 75, interpolation='midpoint')
    iqr_bmi = q3_bmi - q1_bmi
    
    mean_charges = np.mean(data['charges'])
    moda_charges = sts.mode(data['charges'])
    med_charges = np.median(data['charges'])
    std_charges = data['charges'].std()
    raz_charges = data['charges'].max() - data['charges'].min()
    q1_charges = np.percentile(data['charges'], 25, interpolation='midpoint')
    q3_charges = np.percentile(data['charges'], 75, interpolation='midpoint')
    iqr_charges = q3_charges - q1_charges
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 4))
    ax[0][0].hist(mean_bmi,
                label='Среднее',
                bins=5,
                color='yellow',
                edgecolor='black')
    ax[0][0].hist(med_bmi,
                label='Медиана',
                bins=5,
                color='green',
                edgecolor='black')
    ax[0][0].hist(moda_bmi[0],
                label='Мода',
                bins=5,
                color='blue',
                edgecolor='black')
    ax[0][0].legend()

    ax[0][1].hist(mean_charges, label='Среднее', color='yellow', edgecolor='black')
    ax[0][1].hist(med_charges, label='Медиана', color='green', edgecolor='black')
    ax[0][1].hist(moda_charges[0], label='Мода', color='blue', edgecolor='black')
    ax[0][1].legend()

    ax[1][0].hist(raz_bmi,
                label='Размах',
                bins=1,
                color='yellow',
                edgecolor='black')
    ax[1][0].hist(std_bmi,
                label='Ст. отклонение',
                bins=1,
                color='green',
                edgecolor='black')
    ax[1][0].hist(iqr_bmi,
                label='Межкварт. размах',
                bins=1,
                color='blue',
                edgecolor='black')
    ax[1][0].legend()

    ax[1][1].hist(raz_charges,
                label='Размах',
                bins=1,
                color='yellow',
                edgecolor='black')
    ax[1][1].hist(std_charges,
                label='Ст. отклонение',
                bins=1,
                color='green',
                edgecolor='black')
    ax[1][1].hist(iqr_charges,
                label='Межкварт. размах',
                bins=1,
                color='blue',
                edgecolor='black')
    ax[1][1].legend()
    plt.show()

    print(f'''bmi
    Среднее = {mean_bmi}
    Мода = {moda_bmi}
    Медиана = {med_bmi}
    Стандартное отклонение = {std_bmi}
    Размах = {raz_bmi}
    Межквартальный размах = {iqr_bmi}''')
    print(f'''charges
    Среднее = {mean_charges}
    Мода = {moda_charges}
    Медиана = {med_charges}
    Стандартное отклонение = {std_charges}
    Размах = {raz_charges}
    Межквартальный размах = {iqr_charges}''')
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 4))
    ax[0][0].boxplot([data['age']],
                    labels=['age'])
    ax[0][1].boxplot([data['bmi']],
                    labels=['bmi'])
    ax[1][0].boxplot([data['children']],
                    labels=['children'])
    ax[1][1].boxplot([data['charges']],
                    labels=['charges'])
    plt.grid()
    plt.show()
    
    means = [np.mean(randint(data['bmi'].min(), data['bmi'].max(),
                    randint(30, 100))) for i in range(300)]
    plt.hist(means)
    plt.title(
        f'Среднее = {mean(means)}\nСтандартное отклонение = {np.array(means).std()}')
    plt.show()  
    
    print('bmi')
    SE = std_bmi / (data['bmi'].size)**0.5
    print(mean_bmi-1.96*SE, mean_bmi+1.96*SE)
    SE = std_bmi / (data['bmi'].size)**0.5
    print(mean_bmi-2.58*SE, mean_bmi+2.58*SE)

    print('\ncharges')
    SE = std_charges / (data['charges'].size)**0.5
    print(mean_charges-1.96*SE, mean_charges+1.96*SE)
    SE = std_charges / (data['charges'].size)**0.5
    print(mean_charges-2.58*SE, mean_charges+2.58*SE)

if __name__ == '__main__':
    msg_dic = {
        1: 'Задание 1',
        2: 'Задание 2',
        3: 'Задание 3 (4, 5, 6)',
    }
    digit = [1, 2, 3]
    msg = 'Выберите задание или введите СТОП, чтобы завершить программу:'
    while digit != None:
        print(msg)
        choice = input('Выберите задание из списка ' + str(digit) + ': ' ).strip()
        if choice.isdigit():
            choice = int(choice)
            if choice in msg_dic.keys():
                print(msg_dic[choice])
                if choice == 1:
                    f1()
                    digit.remove(choice)
                    msg_dic.pop(choice)
                elif choice == 2:
                    f2()
                    digit.remove(choice)
                    msg_dic.pop(choice)
                elif choice == 3:
                    f3_4_5_6()
                    digit.remove(choice)
                    msg_dic.pop(choice)
                else:
                    print('Ошибка')
            else:
                print('Ошибка')
        if digit == []:
            digit = None
            print('\nВыполнено')
            break
        if choice == 'СТОП':
            print('\nПрограмма завершена')
            break