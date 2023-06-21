import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import pickle
from datetime import datetime, timedelta

from processor2 import * #Importamos todas las clases del archivo processor
from data_injector import *

import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)

import ray

app = Dash(__name__)
ray.init()

app.layout = html.Div([
    html.H1("Selecciona una fecha y hora"),
    dcc.DatePickerSingle(
        id='date-picker',
        placeholder="Selecciona una fecha"
    ),
    dcc.Dropdown(
        id='hour-dropdown',
        options=[{'label': str(i).zfill(2), 'value': i} for i in range(24)],
        placeholder="Selecciona la hora"
    ),
    html.Div(id='output-container')
])


# @app.callback(
#     dash.dependencies.Output('output-container', 'children'),
#     [dash.dependencies.Input('date-picker', 'date'),
#      dash.dependencies.Input('hour-dropdown', 'value')])


def cercanias(end_date):
    fila=0
    start = False
    start2 = False
    start3 = False
    start4 = False
    lista_Lunes = []

    # Crear dataframe "future"
    future = pd.DataFrame({'ds': pd.date_range(start=datetime(2020, 6, 21), end=end_date, freq='H')})

    if start == True and end_date.hour == 0:
        start = False
    elif start2 == True and end_date.hour == 6:
        start2 = False
    elif start3 == True and end_date.hour == 12:
        start3 = False
    elif start4 == True and end_date.hour == 18:
        start4 = False
    else:
        pass

    # Condición de prueba
    if __name__ == '__main__':
        tripsloader = TripsLoader(verbose = True)
        timeseries_o = tripsloader.timeseries_o
        timeseries_o = timeseries_o.loc['2020-07-01':'2020-07-31 23:00:00']

        Trayectos = tripsloader.trayectos
        trayectos = ray.get(Trayectos)
        trayectos = trayectos.loc['2020-06-21':'2020-07-31 23:00:00']
        columnas_o = [columna for columna in trayectos.columns if columna.startswith('2807905')]
        subsetN = trayectos[columnas_o]
        subsetNn = subsetN.copy()

        injector = Injector()
        dfFinal0_5 = injector.dfFinal0_5
        dfFinal6_11 = injector.dfFinal6_11
        dfFinal12_17 = injector.dfFinal12_17
        dfFinal18_23 = injector.dfFinal18_23
    else:
        subsetNn = subsetN.copy()

    future, m = start_model(future)
    forecast = m.predict(future)
    prediction = forecast.iloc[[-1]].copy()
    ds = prediction.loc[:, 'ds']
    yhat = prediction.loc[:, 'yhat']
    prediction = pd.DataFrame({'ds': ds, 'yhat': yhat})
    prediction = prediction.set_index('ds')
    # print(prediction)

    Chamartin_up_Renfe = Input_Estimate_Cercanias(prediction,subsetNn,timeseries_o)

    return Chamartin_up_Renfe


def metro(end_date):
    fila=0
    start = False
    start2 = False
    start3 = False
    start4 = False
    lista_Lunes = []

    # Crear dataframe "future"
    future = pd.DataFrame({'ds': pd.date_range(start=datetime(2020, 6, 21), end=end_date, freq='H')})

    if start == True and end_date.hour == 0:
        start = False
    elif start2 == True and end_date.hour == 6:
        start2 = False
    elif start3 == True and end_date.hour == 12:
        start3 = False
    elif start4 == True and end_date.hour == 18:
        start4 = False
    else:
        pass

    if __name__ == '__main__':
        tripsloader = TripsLoader(verbose = True)
        timeseries_o = tripsloader.timeseries_o
        timeseries_o = timeseries_o.loc['2020-07-01':'2020-07-31 23:00:00']

        Trayectos = tripsloader.trayectos
        trayectos = ray.get(Trayectos)
        trayectos = trayectos.loc['2020-06-21':'2020-07-31 23:00:00']
        columnas_o = [columna for columna in trayectos.columns if columna.startswith('2807905')]
        subsetN = trayectos[columnas_o]
        subsetNn = subsetN.copy()

        injector = Injector()
        dfFinal0_5 = injector.dfFinal0_5
        dfFinal6_11 = injector.dfFinal6_11
        dfFinal12_17 = injector.dfFinal12_17
        dfFinal18_23 = injector.dfFinal18_23
    else:
        subsetNn = subsetN.copy()

    future, m = start_model(future)
    forecast = m.predict(future)
    prediction = forecast.iloc[[-1]].copy()
    ds = prediction.loc[:, 'ds']
    yhat = prediction.loc[:, 'yhat']
    prediction = pd.DataFrame({'ds': ds, 'yhat': yhat})
    prediction = prediction.set_index('ds')
    # print(prediction)

    Chamartin_up_Metro = Input_Estimate_Metro(prediction,subsetNn,timeseries_o)

    return Chamartin_up_Metro

if __name__ == '__main__':
    app.run_server(debug=True)
