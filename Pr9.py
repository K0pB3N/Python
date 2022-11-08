import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import time

data = load_breast_cancer(as_frame=True)


predictors = data.data
target = data.target
target_names = data.target_names

def f1():
    # 1. Найти данные для классификации. Данные в группе повторяться не должны! Предобработать данные, если это необходимо.
    print(predictors.head(5), '\n\nЦелевая переменная\n', target.head(5), '\nИмена классов:\n', target_names)

f1()

def f2():
    # 2. Изобразить гистограмму, которая показывает баланс классов. Сделать выводы
    target.hist(bins=20,  figsize=(7, 7))
    plt.xticks([0, 1])
    plt.show()
f2()

print('\n')

def f3():
    # 3. Разбить выборку на тренировочную и тестовую. Тренировочная для обучения модели, тестовая для проверки ее качества.
    x_train, x_test, y_train, y_test = train_test_split(predictors,
                                                        target,
                                                        train_size=0.8,
                                                        shuffle=True,
                                                        random_state=271)
    print(' Размер для признаков обучающей выборки', x_train.shape, '\n',
        'Размер для признаков тестовой выборки', x_test.shape, '\n',
        'Размер для целевого показателя обучающей выборки', y_train.shape, '\n',
        'Размер для показателя тестовой выборки', y_test.shape, '\n')

    # 4. Применить алгоритмы классификации: логистическая регрессия
    start_time = time.time()
    model = LogisticRegression(random_state=271)
    print(model.fit(x_train, y_train))
    print('\n')

    y_predict = model.predict(x_test)
    print("--- %s seconds ---" % (time.time() - start_time), '\n')
    print(y_predict)
    print(np.array(y_test))
    print('\n')
    
    fig = px.imshow(confusion_matrix(
        y_test, y_predict), text_auto=True)
    fig.update_layout(xaxis_title='Target (Log)', yaxis_title='Prediction')
    fig.show()
    print(classification_report(y_test, y_predict))

    print('\n')
    
    # 4. Применить алгоритмы классификации: SVM
    start_time = time.time()
    param_kernel = ('linear', 'rbf', 'poly', 'sigmoid')
    parameters = {'kernel': param_kernel}
    model = SVC()
    grid_search_svm = GridSearchCV(estimator=model, param_grid=parameters, cv=6)
    print(grid_search_svm.fit(x_train, y_train), '\n')
    best_model = grid_search_svm.best_estimator_
    print('Best model: ', best_model.kernel)
    print('\n')
    svm_preds = best_model.predict(x_test)
    print("--- %s seconds ---" % (time.time() - start_time), '\n')
    print(svm_preds)
    print('\n')
    print(classification_report(svm_preds, y_test))

    fig = px.imshow(confusion_matrix(y_test, svm_preds), text_auto=True)
    fig.update_layout(xaxis_title='Target (SVM)', yaxis_title='Prediction')
    fig.show()

    # 4. Применить алгоритмы классификации: KNN
    start_time = time.time()
    number_of_neighbors = np.arange(3, 10, 25)
    model_KNN = KNeighborsClassifier()
    params = {'n_neighbors': number_of_neighbors}
    print('\n')
    grid_search = GridSearchCV(estimator=model_KNN, param_grid=params, cv=6)
    print(grid_search.fit(x_train, y_train), '\n')
    print('Best score: ',grid_search.best_score_)
    print('\n')
    print('Best estimator: ', grid_search.best_estimator_)
    print('\n')
    knn_preds = grid_search.predict(x_test)
    print("--- %s seconds ---" % (time.time() - start_time), '\n')
    print('\n')
    print(classification_report(knn_preds, y_test))
    
    print('\n')
    fig = px.imshow(confusion_matrix(y_test, knn_preds), text_auto=True)
    fig.update_layout(xaxis_title='Target (KNN)', yaxis_title='Prediction')
    fig.show()
f3()