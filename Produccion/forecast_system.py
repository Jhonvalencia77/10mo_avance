###############################Lunes0_5#######################################
def est_Lunes0_5(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    if (date.dayofweek == 0 and date.hour == 0) or (date.dayofweek == 0 and date.hour == 1) or (date.dayofweek == 0 and date.hour == 2) or (date.dayofweek == 0 and date.hour == 3) or (date.dayofweek == 0 and date.hour == 4) or (date.dayofweek == 0 and date.hour == 5):
        valor = True
        fila += 1
    else:
        valor = False
        fila += 1
    return (valor)
def Regressor0_5_FutureAccUp(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal0_5.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal0_5.loc[fila]
        valor = valor['Accup']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)
def Regressor0_5_LunesBack(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal0_5.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal0_5.loc[fila]
        valor = valor['t-168Mod']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)

#################################Lunes6_11#####################################
def est_Lunes6_11(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    if (date.dayofweek == 0 and date.hour == 6) or (date.dayofweek == 0 and date.hour == 7) or (date.dayofweek == 0 and date.hour == 8) or (date.dayofweek == 0 and date.hour == 9) or (date.dayofweek == 0 and date.hour == 10) or (date.dayofweek == 0 and date.hour == 11):
        valor = True
        fila += 1
    else:
        valor = False
        fila += 1
    return (valor)
def Regressor6_11_FutureAccUp(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal6_11.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal6_11.loc[fila]
        valor = valor['Accup']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)
def Regressor6_11_LunesBack(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal6_11.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal6_11.loc[fila]
        valor = valor['t-168Mod']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)

##################################Lunes12_17####################################
def est_Lunes12_17(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    if (date.dayofweek == 0 and date.hour == 12) or (date.dayofweek == 0 and date.hour == 13) or (date.dayofweek == 0 and date.hour == 14) or (date.dayofweek == 0 and date.hour == 15) or (date.dayofweek == 0 and date.hour == 16) or (date.dayofweek == 0 and date.hour == 17):
        valor = True
        fila += 1
    else:
        valor = False
        fila += 1
    return (valor)
def Regressor12_17_FutureAccUp(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal12_17.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal12_17.loc[fila]
        valor = valor['Accup']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)
def Regressor12_17_LunesBack(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal12_17.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal12_17.loc[fila]
        valor = valor['t-168Mod']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)

####################################Lunes18_23##################################
def Regressor18_23_FutureAccUp(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal18_23.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal18_23.loc[fila]
        valor = valor['Accup']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)
def Regressor18_23_LunesBack(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal18_23.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal18_23.loc[fila]
        valor = valor['t-168Mod']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)

####################################################################################
def Add_y(ds):
    global fila
    global valor
    date = pd.to_datetime(ds)
    fechafuture = dfFinal18_23.ds.loc[fila]
    if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
        valor = dfFinal18_23.loc[fila]
        valor = valor['y']
        fila += 1
    else:
        valor = 0
        fila += 1
    return (valor)

########################################################3
def Verify_Lunes(fechas):
    global lista_Lunes
    for fecha in fechas:
        if fecha.weekday() == 0:
            lista_Lunes.append(fecha)
        else:
           pass
    return (lista_Lunes)

#####################################################
def warm_start_params(m):
    """
    Retrieve parameters from a trained model in the format used to initialize a new Stan model.
    Note that the new Stan model must have these same settings:
        n_changepoints, seasonality features, mcmc sampling
    for the retrieved parameters to be valid for the new model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ['delta', 'beta']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res


def start_model(future):
    global start
    global start2
    global start3
    global start4
    global fila
    global m
    if end_date.hour in [0, 1, 2, 3, 4, 5]:
        fila=0
        future['Lunes0_5'] = future['ds'].apply(est_Lunes0_5)
        fila = 0
        future['Accup'] = future['ds'].apply(Regressor0_5_FutureAccUp)
        fila = 0
        future['t-168Mod'] = future['ds'].apply(Regressor0_5_LunesBack)

        if start == False:
            # Load model
            with open('/home/jonathan/tesis/10mo_avance/models/serialized_model_Lunes0_5.json', 'r') as fin:
                m = model_from_json(fin.read())
            start = True
        else:
            df = future.iloc[:-1].copy()
            fila = 0
            df['y'] = df['ds'].apply(Add_y)
            m2 = Prophet(changepoint_range=0.85,seasonality_prior_scale=0.3,weekly_seasonality=False,daily_seasonality=False,
                    seasonality_mode='additive',changepoint_prior_scale=1.7)
            m2.add_seasonality(name='Weekly', period=7, fourier_order=2,prior_scale=0.8)
            m2.add_seasonality(name='Daily', period=1, fourier_order=4,prior_scale=0.0064)
            m2.add_seasonality(name='Lunes0_5', period=1/4, fourier_order=6, condition_name='Lunes0_5',prior_scale=0.005)
            m2.add_regressor('t-168Mod',mode='additive',prior_scale=0.5)
            m2.add_regressor('Accup',mode='additive',prior_scale=0.05)
            m = m2.fit(df, init=warm_start_params(m))

    elif end_date.hour in [6, 7, 8, 9, 10, 11]:
        fila = 0
        future['Lunes6_11'] = future['ds'].apply(est_Lunes6_11)
        fila = 0
        future['Accup'] = future['ds'].apply(Regressor6_11_FutureAccUp)
        fila = 0
        future['t-168Mod'] = future['ds'].apply(Regressor6_11_LunesBack)

        if start2 == False:
            # Load model
            with open('/home/jonathan/tesis/10mo_avance/models/serialized_model_Lunes6_11.json', 'r') as fin:
                m = model_from_json(fin.read())
            start2 = True
        else:
            df = future.iloc[:-1].copy()
            fila = 0
            df['y'] = df['ds'].apply(Add_y)
            m2 = Prophet(changepoint_range=0.8,seasonality_prior_scale=0.3,weekly_seasonality=False,daily_seasonality=False,
                    seasonality_mode='additive',changepoint_prior_scale=1)   #mcmc_samples=100  Si mejora la predicción->,n_changepoints=100
            m2.add_seasonality(name='Lunes6_11', period=1/4, fourier_order=6, condition_name='Lunes6_11',prior_scale=0.05)
            m2.add_seasonality(name='Weekly', period=7, fourier_order=2,prior_scale=0.1)
            m2.add_seasonality(name='Daily', period=1, fourier_order=4,prior_scale=0.1)
            m2.add_regressor('Accup',mode='additive',prior_scale=1,standardize=False)
            m2.add_regressor('t-168Mod',mode='additive',prior_scale=1)
            m = m2.fit(df, init=warm_start_params(m))


    elif end_date.hour in [12, 13, 14, 15, 16, 17]:
        fila = 0
        future['Lunes12_17'] = future['ds'].apply(est_Lunes12_17)
        fila = 0
        future['Accup'] = future['ds'].apply(Regressor12_17_FutureAccUp)
        fila = 0
        future['t-168Mod'] = future['ds'].apply(Regressor12_17_LunesBack)

        if start3 == False:
            # Load model
            with open('/home/jonathan/tesis/10mo_avance/models/serialized_model_Lunes12_17.json', 'r') as fin:
                m = model_from_json(fin.read())
            start3 = True
        else:
            df = future.iloc[:-1].copy()
            fila = 0
            df['y'] = df['ds'].apply(Add_y)
            m2 = Prophet(changepoint_range=0.8,seasonality_prior_scale=0.3,weekly_seasonality=False,daily_seasonality=False,
                    seasonality_mode='additive',changepoint_prior_scale=0.1)
            m2.add_seasonality(name='Lunes12_17', period=1/4, fourier_order=6, condition_name='Lunes12_17',prior_scale=0.015)
            m2.add_regressor('t-168Mod',mode='additive',prior_scale=0.5)
            m2.add_regressor('Accup',mode='additive',prior_scale=0.5,standardize=False)
            m = m2.fit(df, init=warm_start_params(m))

    elif end_date.hour in [18, 19, 20, 21, 22, 23]:
        fila = 0
        future['Accup'] = future['ds'].apply(Regressor18_23_FutureAccUp)
        fila = 0
        future['t-168Mod'] = future['ds'].apply(Regressor18_23_LunesBack)

        if start4 == False:
            # Load model
            with open('/home/jonathan/tesis/10mo_avance/models/serialized_model_Lunes18_23.json', 'r') as fin:
                m = model_from_json(fin.read())
            start4 = True
        else:
            df = future.iloc[:-1].copy()
            fila = 0
            df['y'] = df['ds'].apply(Add_y)
            m2 = Prophet(changepoint_range=0.8,seasonality_prior_scale=0.3,weekly_seasonality=False,daily_seasonality=False,
                    seasonality_mode='additive',changepoint_prior_scale=0.1)
            m2.add_seasonality(name='Weekly', period=7, fourier_order=2,prior_scale=0.005)
            m2.add_seasonality(name='Daily', period=1, fourier_order=4,prior_scale=0.005)
            m2.add_regressor('t-168Mod',mode='additive',prior_scale=0.5)
            m2.add_regressor('Accup',mode='additive',prior_scale=0.5)
            m = m2.fit(df, init=warm_start_params(m))

    return future, m

def Input_Estimate_Cercanias(prediction,subsetNn,timeseries_o):
    global lista_Lunes
    valores_reemplazo = prediction["yhat"].to_dict()
    valores_reemplazo

    for fecha_hora, valor in valores_reemplazo.items():
        fecha = fecha_hora.date()
        hora = fecha_hora.time().strftime('%H:%M:%S')
        subsetNn.loc[f'{fecha} {hora}', '2807905-2807901'] = valor

    columnas_o = [columna for columna in subsetNn.columns if columna.startswith('2807905')]
    subsetNn['2807905'] = subsetNn.loc[:, columnas_o].sum(axis=1)
    colum_sum = subsetNn.loc[:, '2807905']
    index = subsetNn.index
    chamartin_op = pd.DataFrame({'2807905': colum_sum})
    chamartin_op.index = index

    chamartin_op.index = pd.to_datetime(chamartin_op.index)
    subchamartin_op = chamartin_op[chamartin_op.index == fecha_hora]

    valores_reemplazo = subchamartin_op["2807905"].to_dict()

    for fecha_hora, valor in valores_reemplazo.items():
        fecha = fecha_hora.date()
        hora = fecha_hora.time().strftime('%H:%M:%S')
        timeseries_o.loc[f'{fecha} {hora}', '2807905'] = valor

    tripsloader = TripsLoader(verbose = True)
    print(2)
    ptdata = PassengersDataLoader()

    routeTrip = RouteTrip.remote(tripsloader, ptdata)

    result_renf = routeTrip.get_users_renfe.remote(timeseries_o)
    renfe_up = ray.get(result_renf)

    fecha_lunes_anterior = fecha_hora - pd.DateOffset(days=fecha_hora.dayofweek + 7)
    rango_fechas = pd.date_range(fecha_lunes_anterior, fecha_hora, freq='H')
    lista_Lunes = Verify_Lunes(rango_fechas)
    rango_fechas = pd.DatetimeIndex(lista_Lunes)
    rango_fechas

    Chamartin_up_Renfe = pd.DataFrame({'y': renfe_up['par_5_18']})
    Chamartin_up_Renfe = Chamartin_up_Renfe.loc[rango_fechas]

    return Chamartin_up_Renfe

def Input_Estimate_Metro(prediction,subsetNn,timeseries_o):
    global lista_Lunes
    valores_reemplazo = prediction["yhat"].to_dict()
    valores_reemplazo

    for fecha_hora, valor in valores_reemplazo.items():
        fecha = fecha_hora.date()
        hora = fecha_hora.time().strftime('%H:%M:%S')
        subsetNn.loc[f'{fecha} {hora}', '2807905-2807901'] = valor

    columnas_o = [columna for columna in subsetNn.columns if columna.startswith('2807905')]
    subsetNn['2807905'] = subsetNn.loc[:, columnas_o].sum(axis=1)
    colum_sum = subsetNn.loc[:, '2807905']
    index = subsetNn.index
    chamartin_op = pd.DataFrame({'2807905': colum_sum})
    chamartin_op.index = index

    chamartin_op.index = pd.to_datetime(chamartin_op.index)
    subchamartin_op = chamartin_op[chamartin_op.index == fecha_hora]

    valores_reemplazo = subchamartin_op["2807905"].to_dict()

    for fecha_hora, valor in valores_reemplazo.items():
        fecha = fecha_hora.date()
        hora = fecha_hora.time().strftime('%H:%M:%S')
        timeseries_o.loc[f'{fecha} {hora}', '2807905'] = valor

    tripsloader = TripsLoader(verbose = True)
    ptdata = PassengersDataLoader()

    routeTrip = RouteTrip.remote(tripsloader, ptdata)

    result_met = routeTrip.get_users_metro.remote(timeseries_o)
    metro_up = ray.get(result_met)

    fecha_lunes_anterior = fecha_hora - pd.DateOffset(days=fecha_hora.dayofweek + 7)
    rango_fechas = pd.date_range(fecha_lunes_anterior, fecha_hora, freq='H')
    lista_Lunes = Verify_Lunes(rango_fechas)
    rango_fechas = pd.DatetimeIndex(lista_Lunes)
    rango_fechas

    Chamartin_up_Metro = pd.DataFrame({'y': metro_up['par_4_261']})
    Chamartin_up_Metro = Chamartin_up_Metro.loc[rango_fechas]

    return Chamartin_up_Metro
