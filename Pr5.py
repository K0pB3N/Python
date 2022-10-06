import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import time
import umap.umap_ as umap


def f1():
    data = pd.read_csv('mnist_test.csv')
    class_name = pd.read_csv('type.csv')

    D = data.drop(['label'], axis=1)
    print(D)
    scaler = preprocessing.MinMaxScaler()
    D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)

    n_n = (5, 25, 50)
    v=0
    for i in range(len(n_n)):

        start_time = time.time()
        T = TSNE(n_components=2, perplexity=n_n[i], random_state=123)
        TSNE_features = T.fit_transform(D)
        DATA = D.copy()

        DATA['x'] = TSNE_features[:, 0]
        DATA['y'] = TSNE_features[:, 1]
        v+=1
        print("TSNE", v, "--- %s seconds ---" % (time.time() - start_time))

        fig = plt.figure()
        plt.title(f'perplexity = {n_n[i]}')
        g = sns.scatterplot(x='x',
                        y='y',
                        hue=class_name['label'],
                        data=DATA,
                        palette='bright')
        sns.move_legend(g, "upper right", title='Цифры')
        plt.show()


def f2():
    data = pd.read_csv('mnist_test.csv')
    D = data.drop(['label'], axis=1)
    class_name = pd.read_csv('type.csv')
    DATA = D.copy()
    n_n = (5, 25, 50)
    m_d = (0.1, 0.6)
    um = dict()
    n=0
    for i in range(len(n_n)):
        for j in range(len(m_d)):
            start_time = time.time()
            um[(n_n[i],m_d[j])] = (umap.UMAP(n_neighbors = n_n[i], min_dist=m_d[j], random_state=123).fit_transform(DATA))
            n+=1
            print("UMAP",n, "--- %s seconds" % (time.time() - start_time))
            fig = plt.figure()
            g = sns.scatterplot(x=um[(n_n[i],m_d[j])][:, 0],
                    y=um[(n_n[i],m_d[j])][:, 1],
                    hue=class_name['label'],
                    data=DATA,
                    palette='bright')
            sns.move_legend(g, "upper right", title='Цифры')
            plt.title(f'n_neighbors = {n_n[i]}, min_dist = {m_d[j]}')  
            plt.show()
            
            
def f3():
    data = pd.read_csv('mnist_test.csv')
    D = data.drop(['label'], axis=1)
    DATA = D.copy()
    scaler = preprocessing.MinMaxScaler()
    D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)

    start_time = time.time()
    T = TSNE(n_components=2, perplexity=50, random_state=123)
    TSNE_features = T.fit_transform(D)
    DATA = D.copy()

    DATA['x'] = TSNE_features[:, 0]
    DATA['y'] = TSNE_features[:, 1]
    print("t-SNE --- %s seconds" % (time.time() - start_time))

    um = dict()
    start_time = time.time()
    um[15, 0.1] = (umap.UMAP(n_neighbors=15, min_dist=0.1,
                    random_state=123).fit_transform(DATA))
    print("UMAP --- %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
        msg_dic = {
            1: 'Задание 1',
            2: 'Задание 2',
            3: 'Задание 3',
        }
        digit = [1, 2, 3]
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
                    else:
                        print('Ошибка')
                else:
                    print('Ошибка')
            if digit == []:
                digit = None
                print('\nВыполнено')
                break
