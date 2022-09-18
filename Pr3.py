import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go

def f1():
    print('\n')
    print('Первое задание: выгрузить многомерные данные', '\n')
    data = pd.read_csv('ds_salaries.csv', sep=',')
    print(data)
f1()

print('\n')
def f2():
    print('Второе задание: вывести данные с помощью info() и head()', '\n')
    print('Вывод информации с помощью метода info() и head():', '\n')
    data = pd.read_csv('ds_salaries.csv', sep=',')
    print(data.head(), data.info())
f2()

print('\n')
def f3():
    print ('Третье задание: построить стообчатую диаграмму с помощью bar()', '\n')
    
    data = px.data.gapminder().query("country=='Canada'")
    fig = go.Figure(px.bar(data, x='year', y='pop', color='pop'))
    fig.update_traces(marker=dict(line=dict(color='black', width=2)))
    fig.update_layout(title='Динамика населения Канады',
                      title_y=0.96, 
                      title_x=0.5, 
                      title_font_size=20,
                      title_xanchor='center',
                      title_yanchor='top',
                      xaxis_title='Год',
                      yaxis_title='Население',
                      xaxis_title_font_size=16,
                      xaxis_tickfont_size=14,
                      height = 800,
                      width = None,
                      margin=dict(l=0, r=0, t=60, b=0),
                      yaxis = dict(
                          dtick = 33.4 * 1e6 / 9,
                      )
                        # yaxis = dict(
                        #  tick0 = 5000000,
                        #      dtick = (1.5 * 10**6
                        #  ) / 6,
                     )
    fig.update_xaxes(tickangle=-90)
    fig.show()
f3()

print('\n')

def f4():
    print('Четвертое задание: построить круговую диаграмму', '\n')
    data = px.data.gapminder().query("country=='Canada'")
    graph = go.Figure(px.pie(data, values='year', names='pop', color='pop'))
    graph.update_traces(marker=dict(line=dict(color='black', width=2)))
    graph.update_layout(title='Диаграмма населения Канады',
                        title_y=0.96,
                        title_x=0.46,
                        title_xanchor='center',
                        title_yanchor='top',
                        title_font_size=16,
                        xaxis_title='Год',
                        yaxis_title='Население',
                        xaxis_title_font_size=16,
                        xaxis_tickfont_size=14,
                        yaxis_title_font_size=16,
                        yaxis_tickfont_size=14,
                        height = 700,
                        width = None,
                        margin = dict(l=0, r=0, t=60, b=0))
    graph.update_xaxes(tickangle=-90)
    graph.show()
f4()

print('\n')

def f5():
    print('Пятое задание: построить линейный график', '\n')
    
    data = px.data.gapminder().query("country=='Canada'")
    data2 = px.data.gapminder().query("country=='United States'")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=data['year'], 
                             y=data['pop'], 
                             line=dict(color='crimson'),
                             marker=dict(color='grey'),
                             name='PopulationCanada'))
    
    fig.add_trace(go.Scatter(x=data2['year'],
                             y=data2['pop'],
                             line=dict(color='purple'),
                             marker=dict(color='grey'),
                             name='PopulationUSA'))
    
    fig.update_traces(marker=dict(line=dict(color='black', width=2)))
    fig.update_xaxes(gridwidth=2, gridcolor='azure')
    fig.update_yaxes(gridwidth=2, gridcolor='azure')
    
    fig.update_layout(yaxis=dict(dtick = (33.6 * 10**7) / 6),
                      title='Динамика населения Канады и США',
                      title_y=0.96,
                      title_x=0.45,
                      title_xanchor='center',
                      title_yanchor='top',
                      title_font_size=16,
                      xaxis_title='Год',
                      yaxis_title='Население',
                      xaxis_title_font_size=16,
                      xaxis_tickfont_size=14,
                      yaxis_title_font_size=16,
                      yaxis_tickfont_size=14,
                      height=900,
                      width=None,
                      margin=dict(l=0, r=500, t=60, b=0))
    fig.update_layout(legend=dict(
        title_text='Legend',
        yanchor='bottom',
        y=0,
        xanchor='left',
        x=-0.3
    ))
    fig.show()
f5()

print('\n')

def f6():
    print('Шестое задание: построить ящик с усами', '\n')
    data = px.data.gapminder().query("country=='Canada'")
    graph = go.Figure(
        go.Box(x=data['year'], 
               y=data['lifeExp'],
               line=dict(color='crimson'), 
               marker=dict(color='darkblue')))
    graph.update_traces(marker=dict(line=dict(color='black', width=2)))
    graph.update_xaxes(gridwidth=2, gridcolor='azure')
    graph.update_yaxes(gridwidth=2, gridcolor='azure')
    graph.update_layout(title='График распределения средней продолжительности жизни в Канаде',
                        yaxis_title='Средняя продолжительность жизни',
                        xaxis_title='Год',
    )
    graph.show()
f6()

print('\n')

def f7():
    print('Седьмое задание: построение графиков с помощью matplotlib', '\n')
    data = px.data.gapminder().query("country=='Canada'")
    graph = plt.figure(figsize=(9, 7))
    ax = graph.add_subplot()
    plt.grid(True)
    plt.title('Диаграмма количества населения в Канаде', fontsize=16)
    plt.xlabel('Год', fontsize=12)
    plt.ylabel('Население', fontsize=14)
    plt.plot(data['year'], 
             data['pop'], 
             marker='.', 
             color='crimson',
             markerfacecolor='white', 
             markeredgecolor='black', 
             markersize=10)
             
    ax.patch.set_facecolor('azure')
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    plt.show()
f7()

print('\n')

def f8():
    print('Седьмое (часть 2) задание: построение графиков с помощью matplotlib', '\n')
    data = px.data.gapminder().query("country=='Canada'")
    graph = plt.figure(figsize=(6, 4))
    ax = graph.add_subplot()
    vals = data['pop'].values.tolist()
    ax.pie(vals, autopct='%.2f', shadow=True)
    ax.grid()
    plt.show()
f8()

print('\n')

def f9():
    print('Седьмое задание (часть 3): построение графиков с помощью matplotlib', '\n')
    data = px.data.gapminder().query("country=='Canada'")
    graph = plt.figure(figsize=(9, 4))
    ax = graph.add_subplot()
    x = data['year']
    y = data['pop']
    ax.bar(x, y)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    ax.grid()
    plt.show()
f9()