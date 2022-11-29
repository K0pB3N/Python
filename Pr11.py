import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import catboost as cb
import time

# Загрузка данных и разделение на предикторы и целевую переменную
data = load_breast_cancer(as_frame=True)
predictors = data.data
target = data.target

def f1(): # train_test_split
    # Разделение на обучающую и тестовую выборки
    A_train, A_test, y_train, y_test = train_test_split(predictors,
                                                        target,
                                                        train_size=0.8)
    # Обучение модели случайного леса
    random_forest = RandomForestClassifier(max_depth=15, min_samples_split=10).fit(
        A_train, y_train)
    
    
    y_preds_d = random_forest.predict(A_train)
    print('F1 мера для тренировочных данных\n', f1_score(
        y_preds_d, y_train, average='macro'))
    y_preds = random_forest.predict(A_test)
    print('F1 мера для тестовых данных\n', f1_score(y_preds, y_test, average='macro'))

    def f2(): # Random Forest Classifier
        random_forest = RandomForestClassifier()
        params_grid = {
            'max_depth': [12, 18],
            'min_samples_leaf': [3, 10],
            'min_samples_split': [6, 12]
        }
        start_time = time.time()
        # Поиск по сетке
        grid_search_random_forest = GridSearchCV(estimator=random_forest,
                                                param_grid=params_grid,
                                                scoring='f1_macro',
                                                cv=4)
        print(grid_search_random_forest.fit(A_train, y_train))

        best_model = grid_search_random_forest.best_estimator_
        print(best_model)
        time1 = time.time() - start_time

        y_preds_d = best_model.predict(A_train)
        print('\n')
        print('F1 мера для тренировочных данных', f1_score(
        y_preds_d, y_train, average='macro'))

        y_preds = best_model.predict(A_test)
        score1 = f1_score(y_preds, y_test, average='macro')
        print('F1 мера для тестовых данных', score1)

        def f3(): # CatBoost 
            # Обучение модели CatBoost
            start_time = time.time()
            model_catboost_clf = cb.CatBoostClassifier(iterations=3000,
                                                    task_type='GPU',
                                                    devices='0')
            model_catboost_clf.fit(A_train, y_train)
            time2 = time.time() - start_time

            # Предсказание на тестовой выборке
            y_preds_t = model_catboost_clf.predict(A_train, task_type='CPU')
            print('F1 мера для тренировочных данных', f1_score(
                y_preds_t, y_train, average='macro'))

            # Предсказание на тестовой выборке
            y_preds = model_catboost_clf.predict(A_test, task_type='CPU')
            score2 = f1_score(y_preds, y_test, average='macro')
            print('F1 мера для тестовых данных', score2)

            def f4(): # Boosting + Bagging - time and efficiency
                df = pd.DataFrame({
                    'Algorithm': ['Баггинг', 'Бустинг'],
                    'time': [time1, time2],
                    'Efficiency': [score1, score2]
                })
                print(df)
            f4()
        f3()
    f2()
f1()