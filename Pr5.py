import umap.umap_ as umap
import pandas as pd
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time

def f1():
    data = pd.read_csv('vgsales.csv')
    leg = pd.read_csv('vgsales.csv')
    data
    D = data.drop(['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher',
                'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)
    scaler = preprocessing.MinMaxScaler()
    n = [5, 25, 50]
    while n != None:
        if n == []:
            print('n is empty')
            break
        print('Введите число перплексии: ' + str(n))
        q = input()
        if q == 'СТОП':
            print('Выход из программы. \n')
            break
        else:
            q = int(q)
            if q == 5:
                D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)
                start_time = time.time()
                T = TSNE(n_components=2, perplexity=q, random_state=123)
                TSNE_features = T.fit_transform(D)
                DATA = D.copy()

                DATA['x'] = TSNE_features[:, 0]
                DATA['y'] = TSNE_features[:, 1]
                print("--- %s seconds ---" % (time.time() - start_time))


                fig = plt.figure(figsize=(15, 10))
                plt.title(f'perplexity = 5')
                g = sns.scatterplot(x='x',
                                    y='y',
                                    hue=leg['Genre'],
                                    data=DATA,
                                    palette='bright')
                sns.move_legend(g, "upper right", title='Genre')
                plt.show()
                n.remove(q)
                
            elif q == 25:
                D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)
                start_time = time.time()
                T = TSNE(n_components=2, perplexity=q, random_state=123)
                TSNE_features = T.fit_transform(D)
                DATA = D.copy()

                DATA['x'] = TSNE_features[:, 0]
                DATA['y'] = TSNE_features[:, 1]
                print("--- %s seconds ---" % (time.time() - start_time))

                fig = plt.figure(figsize=(15, 10))
                plt.title(f'perplexity = 25')
                g = sns.scatterplot(x='x',
                                    y='y',
                                    hue=leg['Genre'],
                                    data=DATA,
                                    palette='bright')
                sns.move_legend(g, "upper right", title='Genre')
                plt.show()
                n.remove(q)
                
            elif q == 50:
                D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)
                start_time = time.time()
                T = TSNE(n_components=2, perplexity=q, random_state=123)
                TSNE_features = T.fit_transform(D)
                DATA = D.copy()

                DATA['x'] = TSNE_features[:, 0]
                DATA['y'] = TSNE_features[:, 1]
                print("--- %s seconds ---" % (time.time() - start_time))

                fig = plt.figure(figsize=(15, 10))
                plt.title(f'perplexity = 50')
                g = sns.scatterplot(x='x',
                                    y='y',
                                    hue=leg['Genre'],
                                    data=DATA,
                                    palette='bright')
                sns.move_legend(g, "upper right", title='Genre')
                plt.show()
                n.remove(q)
                
def f2():
    data = pd.read_csv('vgsales.csv')
    leg = pd.read_csv('vgsales.csv')
    data
    D = data.drop(['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher',
                'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)
    scaler = preprocessing.MinMaxScaler()
    D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)

    start_time = time.time()
    T = TSNE(n_components=2, perplexity=5, random_state=123)
    TSNE_features = T.fit_transform(D)
    DATA = D.copy()

    n_n = (5, 25, 50)
    m_d = (0.1, 0.6)
    um = dict()
    for i in range(len(n_n)):
        for j in range(len(m_d)):
            um[(n_n[i], m_d[j])] = (umap.UMAP(n_neighbors=n_n[i],
                                            min_dist=m_d[j], random_state=123).fit_transform(DATA))
            fig = plt.figure(figsize=(20, 15))
            g2 = sns.scatterplot(x=um[(n_n[i], m_d[j])][:, 0],
                                y=um[(n_n[i], m_d[j])][:, 1],
                                hue=leg['Genre'],
                                data=DATA,
                                palette='bright')
            plt.title(f'n_neighbors = {n_n[i]}, min_dist = {m_d[j]}')
            sns.move_legend(g2, "upper right", title='Genre')
            plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
    
print('\n')


def f3():
    data = pd.read_csv('vgsales.csv')
    D = data.drop(['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher',
                'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)


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
