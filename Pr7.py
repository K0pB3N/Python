import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas import DataFrame

def f1():
    street = np.array([80,98,75,91,78])
    garage = np.array([100,82,105,89,102])
    day = np.array(['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница'])
    print("1) Два вектора: ", np.corrcoef(street,garage))
    print('\n')
    def f2():
        print("2) Корреляция по Пирсону: ", np.corrcoef(street,garage)[0,1])
    f2()
    def f3():
        plt.grid(True)
        plt.title('Диаграмма рассеяния')
        plt.xlabel('Число автомобилей')
        plt.ylabel('День недели')
        plt.scatter(street, day,  marker='o', color='crimson')
        plt.scatter(garage, day,  marker='x', color='crimson')
        print("3) Диаграмма рассеяния: ")
        plt.show()
    f3()
f1()

def f4(): 
    data = pd.read_csv('bitcoin.csv')
    print("4) Загрузить данные из файла “bitcoin.csv”")
    print(data)
    print('\n')
    def f5(): 
        projection = 14
        data['predict'] = data['close'].shift(-projection)
        print("5) Скрыть последние 14 дней.: ")
        print(data)
        print('\n')
    f5()
    def f6():
        projection = 14
        x = DataFrame(data, columns=['close'])
        y = DataFrame(data, columns=['predict'])
        x = np.array(x, type(float))[:-projection]
        y = np.array(y, type(float))[:-projection]
        print("5) Произвести нормализацию для нормального среза." +
              "\n" + "6) Сделать срез по 14 дням: ")
        print(y)
        print('\n')
    f6()
    def f7():
        projection = 14
        x = DataFrame(data, columns=['close'])
        y = DataFrame(data, columns=['predict'])
        x = np.array(x, type(float))[:-projection]
        y = np.array(y, type(float))[:-projection]
        regression = LinearRegression()
        regression.fit(x, y)
        print("7) Поcтроить линейную регрессию: ")
        print(regression.predict(x))
        print('\n')
        print("8) Вывести угол наклона и y-перехват: ")
        print(regression.coef_)
        print('\n')
        print(regression.intercept_)
        print('\n')
    f7()
    def f8():
        projection = 14
        x = DataFrame(data, columns=['close'])
        y = DataFrame(data, columns=['predict'])
        x = np.array(x, type(float))[:-projection]
        y = np.array(y, type(float))[:-projection]
        regression = LinearRegression()
        regression.fit(x, y)
        print("9) Предсказать стоимость криптовалюты за последние 14 дней с помощью функции “predict”: ")
        print(regression.predict(data[['close']][-projection:]))
        print('\n')
        print("10) Определить точность прогнозируемой цены закрытия с помощью функции “ score”: ")
        print(regression.score(x, y))
    f8()
f4()

print('\n')
print("11) Сравнить скрытые значения с предсказанными. Сделать вывод о том, насколько они схожи.")
print('\n')

def f9():
    data = pd.read_csv('housePrice.csv')
    print("12) Загрузить данные из файла “housePrice.csv”" + "\n")
    print(data)
    print('\n')
    def f10():
        data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
        data['Price(USD)'] = pd.to_numeric(data['Price(USD)'], errors='coerce')
        x = data['Area']
        y = data['Price(USD)']
        print("13) Произвести предобработку. \n" +
              "14) Реализовать линейную регрессию вручную, без использования библиотеки. За основу взять два признака: “Area” и “Price(USD)”: \n")
        print(x)
        print('\n')
    f10()
    def f11():
        data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
        data['Price(USD)'] = pd.to_numeric(data['Price(USD)'], errors='coerce')
        x = data['Area']
        y = data['Price(USD)']
        
        n = np.size(x)  # количество точек
        m_x = np.mean(x)  # среднее значение векторов x и y
        m_y = np.mean(y)
        # вычисление перекрестного отклонения и отклонения около x
        SS_xy = np.sum(y*x) - n*m_y*m_x
        SS_xx = np.sum(x*x) - n*m_x*m_x
        b_1 = SS_xy / SS_xx  # вычисление коэффов регрессии
        b_0 = m_y - b_1*m_x
        print("15) Вывести угол наклона и y-перехват: \n")
        print(f'Коэффициенты: наклон линии регрессии = {b_1}, y-перехват = {b_0}')
        print('\n')
    f11()
    def f12():
        data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
        data['Price(USD)'] = pd.to_numeric(data['Price(USD)'], errors='coerce')
        x = data['Area']
        y = data['Price(USD)']
        n = np.size(x)  # количество точек
        m_x = np.mean(x)  # среднее значение векторов x и y
        m_y = np.mean(y)
        # вычисление перекрестного отклонения и отклонения около x
        SS_xy = np.sum(y*x) - n*m_y*m_x
        SS_xx = np.sum(x*x) - n*m_x*m_x
        b_1 = SS_xy / SS_xx  # вычисление коэффов регрессии
        b_0 = m_y - b_1*m_x
        plt.scatter(x, y, color='m', marker='o', s=30, alpha=0.5)
        y_pred = b_0 + b_1 * x  # пронозируемый вектор
        plt.plot(x, y_pred, color='g')
        plt.xlabel('x')
        plt.ylabel('y')
        print("16) Визуализировать линию регрессии на диаграмме рассеяния. Изменить параметр плотности с помощью команды “alpha”: \n")
        plt.show()
    f12()
f9()
