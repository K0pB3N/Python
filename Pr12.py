import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apriori_python import apriori
from apyori import apriori as apri
from efficient_apriori import apriori as eff_apr
from fpgrowth_py import fpgrowth
import time

data_market = pd.read_csv('Market_Basket_Optimisation.csv')

def f1():
    print(data_market)
    print('\n')
    print(data_market.info())
    
print('\n')

def f2():
    print(data_market.stack().value_counts(normalize=True)[:20].plot(
        kind='bar'))  # относительная частота
    plt.show()
print('\n')

def f3():
    print(data_market.stack().value_counts().apply(lambda item: item /
                                      data_market.shape[0])[:20].plot(kind='bar'))  # фактическая частота
    plt.show()


print('\n')


def f4():
    print(
    '''
    В большинстве библиотек, которые реализуют алгоритмы поиска ассоциативных правил в качестве входных значений необходимо подавать список транзакций, 
    то есть список списков, поэтому нам необходимо преобразовать наш датасет в такой формат. 
    При этом необходимо контролировать пустые значения в данных (NaN), их добавлять в список не надо.
    '''
    )
    
    print('\n')
    
    transactions = []
    for i in range(0, data_market.shape[0]):
        row = data_market.iloc[i].dropna().tolist()
        transactions.append(row)
        
    print('\n')
    
    print(
    '''Также выведем первый элемент и первый список, и сравним с нашим датасетом, чтобы удостоверится, что преобразование прошло удачно'''
    )
    print('\n')
    
    print(f'{transactions[0][0]}\n{transactions[0]}')

    print('\n')
    
    print(
    '''
    minSup – это минимальная поддержка. 
    Значение поддержки меняется от 0 (когда условие и следствие не встречаются вместе ни в одной транзакции) 
    до 1 (когда условие и следствие во всех транзакциях появляются совместно). 

    minConf – минимальная достоверность. 
    Это показатель, характеризующий уверенность в том, что ассоциация A → B является ассоциативным правилом. 
    То есть предположение о том, что появление события A влечёт за собой появление события B, является достаточно достоверным.
    '''
    )
    
    print('\n')
    
    t = []
    start = time.perf_counter()

    t1, rules = apriori(transactions, minSup=0.02, minConf=0.3)
    time1 = (time.perf_counter()-start)
    t.append(time1)
    print(rules)
    
    print('\n')
    
    # apyori
    print(
    '''
    Здесь принимаются те же аргументы, что и в прошлой библиотеке, 
    но необходимо указать минимальный лифт чуть больше 1, чтобы исключить вывод независимых правил
    '''
    )
    
    print('\n')
    
    start = time.perf_counter()
    rules = apri(transactions=transactions,
                min_support=0.02,
                min_confidence=0.3,
                min_lift=1.0001)
    results = list(rules)
    time1 = (time.perf_counter() - start)
    t.append(time1)
    for result in results:
        for subset in result[2]:
            print(subset[0], subset[1])
            print('Support: {0}; Confidence: {1}, Lift: {2};'.format(
                result[1], subset[2], subset[3]))
            print()
            
    print('\n')
    
    # eff_apriori
    start = time.perf_counter()

    itemsets, rules = eff_apr(transactions, min_support=0.02, min_confidence=0.3)
    time1 = (time.perf_counter() - start)
    t.append(time1)
    for i in range(0, len(rules)):
        print(rules[i])
    
    print('\n')
    
    # fpgrowth_py
    start = time.perf_counter()
    itemsets, rules = fpgrowth(transactions, minSupRatio=0.02, minConf=0.3)
    time1 = (time.perf_counter() - start)
    t.append(time1)
    for i in range(0, len(rules)):
        print(rules[i])
        
    print('\n')
    
    print('Время выполнения apriori: ', t[0], '\n')
    print('Время выполнения apriori 2: ', t[1], '\n')
    print('Время выполнения efficient_apriori: ', t[2], '\n')
    print('Время выполнения fpgrowth: ', t[3], '\n')
    plt.bar(['apriori', 'apriori 2', 'efficient_apriori', 'fpgrowth'], t)
    plt.show()

print('\n')

data_data = pd.read_csv('data.csv')

def f5():
    print(data_data)
    print('\n')
    print(data_data.info())
    print('\n')

def f6():
    print(data_data.stack().value_counts(normalize=True)[:20].plot(
        kind='bar'))  # относительная частота
    plt.show()
    
    print('\n')

def f7():
    print(data_data.stack().value_counts().apply(lambda item: item /
                                      data_data.shape[0])[:20].plot(kind='bar'))  # фактическая частота
    plt.show()
    
    print('\n')

def f8():
    print(
    '''
    В большинстве библиотек, которые реализуют алгоритмы поиска ассоциативных правил в качестве входных значений необходимо подавать список транзакций, 
    то есть список списков, поэтому нам необходимо преобразовать наш датасет в такой формат. 
    При этом необходимо контролировать пустые значения в данных (NaN), их добавлять в список не надо.
    '''
    )
    
    print('\n')
    
    transactions = []
    for i in range(0, data_data.shape[0]):
        row = data_data.iloc[i].dropna().tolist()
        transactions.append(row)
        
    print(
    '''Также выведем первый элемент и первый список, и сравним с нашим датасетом, чтобы удостоверится, что преобразование прошло удачно''')
    
    print('\n')
    
    print(f'{transactions[0][0]}\n{transactions[0]}')

    print('\n')
    
    # apriori_python
    
    print(
    '''
    minSup – это минимальная поддержка. 
    Значение поддержки меняется от 0 (когда условие и следствие не встречаются вместе ни в одной транзакции) 
    до 1 (когда условие и следствие во всех транзакциях появляются совместно). 

    minConf – минимальная достоверность. 
    Это показатель, характеризующий уверенность в том, что ассоциация A → B является ассоциативным правилом. 
    То есть предположение о том, что появление события A влечёт за собой появление события B, является достаточно достоверным.
    '''
    )
    
    print('\n')
    
    t = []
    start = time.perf_counter()

    t1, rules = apriori(transactions, minSup=0.02, minConf=0.3)
    time1 = (time.perf_counter() - start)
    t.append(time1)
    print(rules)
    
    print('\n')

    # apyori
    print(
    '''
    Здесь принимаются те же аргументы, что и в прошлой библиотеке, 
    но необходимо указать минимальный лифт чуть больше 1, чтобы исключить вывод независимых правил
    '''
    )

    print('\n')
    
    start = time.perf_counter()
    rules = apri(transactions=transactions,
                min_support=0.02,
                min_confidence=0.3,
                min_lift=1.0001)
    results = list(rules)
    time1 = (time.perf_counter() - start)
    t.append(time1)
    for result in results:
        for subset in result[2]:
            print(subset[0], subset[1])
            print('Support: {0}; Confidence: {1}, Lift: {2};'.format(
                result[1], subset[2], subset[3]))
            print()

    print('\n')
    
    # eff_apriori
    start = time.perf_counter()

    itemsets, rules = eff_apr(transactions, min_support=0.02, min_confidence=0.3)
    time1 = (time.perf_counter() - start)
    t.append(time1)
    for i in range(0, len(rules)):
        print(rules[i])

    print('\n')
    
    # fpgrowth_py
    start = time.perf_counter()
    itemsets, rules = fpgrowth(transactions, minSupRatio=0.02, minConf=0.3)
    time1 = (time.perf_counter() - start)
    t.append(time1)
    for i in range(0, len(rules)):
        print(rules[i])
        
    print('\n')
    
    print('Время выполнения apriori: ', t[0], '\n')
    print('Время выполнения apriori 2: ', t[1], '\n')
    print('Время выполнения efficient_apriori: ', t[2], '\n')
    print('Время выполнения fpgrowth: ', t[3], '\n')
    plt.bar(['apriori', 'apriori 2', 'efficient_apriori', 'fpgrowth'], t)
    plt.show()


if __name__ == '__main__':
    msg_dic = {
        1: 'Задание 1',
        2: 'Задание 2',
        3: 'Задание 3',
        4: 'Задание 4',
        5: 'Задание 5',
        6: 'Задание 6',
        7: 'Задание 7',
        8: 'Задание 8',
        0: 'Выход'
    }
    digit = [1, 2, 3, 4, 5, 6, 7, 8]
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
                elif choice == 8:
                    f8()
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
