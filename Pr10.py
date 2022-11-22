import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
# from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
import time

# data = load_iris(as_frame=True).data

data = load_breast_cancer(as_frame=True).data
def f1():
    print(data)
    
f1()

def f2():
    models = []
    score1 = []
    score2 = []
    for i in range(2, 10):
        model = KMeans(n_clusters=i, random_state=123, init='k-means++').fit(data)
        models.append(model)
        score1.append(model.inertia_)
        score2.append(silhouette_score(data, model.labels_))

    plt.grid()
    plt.plot(np.arange(2, 10), score1, marker='o')
    plt.show()

    plt.grid()
    plt.plot(np.arange(2, 10), score2, marker='o')
    plt.show()
    print('Коэффициент силуэта максимален при k = 2')
f2()

def f3():
    
    start_time = time.time()
    model1 = KMeans(n_clusters=2, random_state=123, init='k-means++')
    model1.fit(data)
    time1 = time.time() - start_time
    print('Смотрим координаты центров кластеров в пространстве:')
    print(model1.cluster_centers_)

    labels = model1.labels_
    data['Claster'] = labels
    data['Claster'].value_counts()
    
    def f4():
        fig = go.Figure(data=[
            go.Scatter3d(x=data['mean radius'],
                         y=data['mean texture'],
                         z=data['mean perimeter'],
                        mode='markers',
                        marker_color=data['Claster'],
                        marker_size=4)
        ])
        fig.show()
    f4()
    
    start_time = time.time()
    model2 = AgglomerativeClustering(2, compute_distances=True)
    model2.fit(data)
    time2 = time.time() - start_time
    
    labels = model2.labels_
    data['Claster'] = labels
    data['Claster'].value_counts()

    def f5():
        fig = go.Figure(data=[
            go.Scatter3d(x=data['mean radius'],
                         y=data['mean texture'],
                         z=data['mean perimeter'],
                        mode='markers',
                        marker_color=data['Claster'],
                        marker_size=4)
        ])
        fig.show()
    f5()
    
    start_time = time.time()
    model3 = DBSCAN(eps=16, min_samples=12).fit(data)
    time3 = time.time() - start_time

    labels = model3.labels_
    data['Claster'] = labels
    print(data['Claster'].value_counts())
    print('\n')
    print('Номер кластера «-1» – это объекты, которые алгоритм выделил как шумовые, то есть это точки, в окрестности которых нет основных точек и всего точек меньше N')

    def f6():
        fig = go.Figure(data=[
            go.Scatter3d(x=data['mean radius'],
                         y=data['mean texture'],
                         z=data['mean perimeter'],
                        mode='markers',
                        marker_color=data['Claster'],
                        marker_size=4)
        ])
        fig.show()
    f6()
    
    df = pd.DataFrame({
        'Algorithm': ['KMeans', 'AgglomerativeClustering', 'DBSCAN'],
        'time': [time1, time2, time3]
    })
    print(df)
f3()