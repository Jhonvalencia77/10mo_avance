import ray
import asyncio
import os
from data_loader import * #Importamos todas las clases del archivo data_loader

class RouteTrip:
    __COEF_BASENAME = PassengersDataLoader.BASEPATH + "%s_coef.csv" #data/pt_data/%s_coef.csv

    def __init__(self,tripsloader,ptdata):
        #propiedad del objeto que recibe el parámetro tripsloader
        self.tripsLoader = tripsloader
        # Propiedad para elegir el archivo con la clase PassengersDataLoader
        self.ptdata = ptdata
        #Propiedad que contiene el archivo routes.csv
        self.routes = tripsloader.routesLoader.routes.set_index("stop_id")

    def get_users_renfe(self, timeseries_o,timeseries_d,max_precision = True):
        #  Filtramos las filas del archivo routes.csv mediante el "stop_id" del archivo up_down_bystop.csv
        renfe_stops = self.routes.loc[self.ptdata.renfe_users_bystop[self.ptdata.renfe_users_bystop.stop_id.notna()].stop_id]
        # Se eliminan las filas con indice duplicado
        renfe_stops = renfe_stops[~renfe_stops.index.duplicated(keep='first')]

        # seleccionamos un subconjunto de columnas de timeseries_o donde están los distritos con paradas de renfe
        subtimeseries_o = timeseries_o[renfe_stops.stop_district.unique()]
        # Se filtran las filas que se encuentren dentro del rango
        subtimeseries_o = subtimeseries_o.between_time("6:00", "0:00")

        subtimeseries_d = timeseries_d[renfe_stops.stop_district.unique()]
        subtimeseries_d = subtimeseries_d.between_time("6:00", "0:00")

        # Td es el dataframe renfe_monthly_data.csv modificado en "ds" (Usuarios reales)
        Td = self.ptdata.renfe_monthly_data

        # Calculamos el número de usuarios por estación
        up,down = self.compute_users_method1(renfe_stops,subtimeseries_o,subtimeseries_d,Td,'cercanias')
        if max_precision:
            up2,down2 = self.compute_users_method2(renfe_stops,subtimeseries_o,subtimeseries_d,self.ptdata.renfe_users_bystop,Td,'cercanias')
            up = (up + up2)/2
            down = (down + down2)/2

        return up,down

    def get_users_metro(self,timeseries_o,timeseries_d, max_precision = True):
        metro_stops = self.routes.loc[self.ptdata.metro_users_bystop[self.ptdata.metro_users_bystop.stop_id.notna()].stop_id]
        metro_stops = metro_stops[~metro_stops.index.duplicated(keep='first')]

        subtimeseries_o = timeseries_o[metro_stops.stop_district.unique()]
        subtimeseries_o = subtimeseries_o.between_time("6:00", "0:00")

        subtimeseries_d = timeseries_d[metro_stops.stop_district.unique()]
        subtimeseries_d = subtimeseries_d.between_time("6:00", "0:00")

        Td = self.ptdata.metro_daily_data

        up, down = self.compute_users_method1(metro_stops, subtimeseries_o, subtimeseries_d, Td,'metro')
        if max_precision:
            up2, down2 = self.compute_users_method2(metro_stops, subtimeseries_o, subtimeseries_d,
                                                    self.ptdata.metro_users_bystop, Td,'metro')
            up = (up + up2) / 2
            down = (down + down2) / 2

        return up, down

    def compute_users_method1(self,stops,subtimeseries_o,subtimeseries_d,Td,service):
        if service == 'cercanias':
            coef = self.compute_p(self.tripsLoader,self.ptdata.renfe_users_bystop,service)
        elif service == 'metro':
            coef = self.compute_p(self.tripsLoader,self.ptdata.metro_users_bystop,service)

        up = pd.DataFrame({}, index=subtimeseries_o.index)
        down = pd.DataFrame({}, index=subtimeseries_d.index)
        for stop_id in stops.index:
            up[stop_id] = subtimeseries_o[stops.loc[stop_id].stop_district] * coef.loc[stop_id].p_o
            down[stop_id] = subtimeseries_d[stops.loc[stop_id].stop_district] * coef.loc[stop_id].p_d
        p = []
        if service == 'cercanias':
            for time_period in Td.index:
                up_p = up[(up.index.month == time_period.month) &
                                                       (up.index.year == time_period.year)]
                down_p = down[(down.index.month == time_period.month) &
                                                       (down.index.year == time_period.year)]

                p_co = up_p.sum().sum()/(Td.loc[time_period].iloc[0]*1e6)
                p_cd = down_p.sum().sum()/(Td.loc[time_period].iloc[0]*1e6)
                p.append((p_co + p_cd)/2)

                up[(up.index.month == time_period.month) & (up.index.year == time_period.year)] = up[(up.index.month == time_period.month) & (up.index.year == time_period.year)]/p_co
                down[(down.index.month == time_period.month) & (down.index.year == time_period.year)] = down[(down.index.month == time_period.month) & (down.index.year == time_period.year)]/p_cd

        elif service == 'metro':
            for time_period in Td.index:
                up_p = up[up.index.date == time_period.date()]
                down_p = down[down.index.date == time_period.date()]

                p_co = up_p.sum().sum() / (Td.loc[time_period].iloc[0])
                p_cd = down_p.sum().sum() / (Td.loc[time_period].iloc[0])
                p.append((p_co + p_cd) / 2)

                up[up.index.date == time_period.date()] = up[up.index.date == time_period.date()] / p_co
                down[down.index.date == time_period.date()] = down[down.index.date == time_period.date()] / p_cd

        return up, down

    def compute_users_method2(self,stops,subtimeseries_o,subtimeseries_d,usersbystop,Td,service):

        beta = usersbystop
        stops = stops["stop_district"]

        beta = beta[beta.stop_id.notna()]
        beta.up = beta.up / beta.up.sum()
        beta.down = beta.down / beta.down.sum()

        up = pd.DataFrame({})
        down = pd.DataFrame({})
        if service == "cercanias":
            for time_period in Td.index:
                subtimeseries_op = subtimeseries_o[(subtimeseries_o.index.month == time_period.month) &
                                                   (subtimeseries_o.index.year == time_period.year)]

                subtimeseries_dp = subtimeseries_d[(subtimeseries_d.index.month == time_period.month) &
                                                   (subtimeseries_d.index.year == time_period.year)]
                # subtimeseries_o[subtimeseries_o.index.weekday >= 5] = subtimeseries_o[subtimeseries_o.index.weekday >= 5] * 0
                alpha_o = subtimeseries_op / subtimeseries_op.sum()
                alpha_d = subtimeseries_dp / subtimeseries_dp.sum()

                # subtimeseries_d[subtimeseries_d.index.weekday >= 5] = subtimeseries_d[subtimeseries_d.index.weekday >= 5] * 0

                up_aux = pd.DataFrame({}, index=subtimeseries_op.index)
                down_aux = pd.DataFrame({}, index=subtimeseries_dp.index)
                for stop_id in beta.stop_id:

                    up_aux[stop_id] = alpha_o[stops.loc[stop_id]] * Td.loc[time_period].iloc[0] * 1e6 * \
                                      beta[beta.stop_id == stop_id].up.iloc[0]
                    down_aux[stop_id] = alpha_d[stops.loc[stop_id]] * Td.loc[time_period].iloc[0] * 1e6 * \
                                        beta[beta.stop_id == stop_id].down.iloc[0]

                up = up.append(up_aux)
                down = down.append(down_aux)

        elif service == 'metro':
            for time_period in Td.index:
                subtimeseries_op = subtimeseries_o[subtimeseries_o.index.date == time_period.date()]
                subtimeseries_dp = subtimeseries_d[subtimeseries_d.index.date == time_period.date()]

                alpha_o = subtimeseries_op / subtimeseries_op.sum()
                alpha_d = subtimeseries_dp / subtimeseries_dp.sum()

                up_aux = pd.DataFrame({}, index=subtimeseries_op.index)
                down_aux = pd.DataFrame({}, index=subtimeseries_dp.index)
                for stop_id in beta.stop_id:
                    up_aux[stop_id] = alpha_o[stops.loc[stop_id]] * Td.loc[time_period].iloc[0] * \
                                      beta[beta.stop_id == stop_id].up.iloc[0]
                    down_aux[stop_id] = alpha_d[stops.loc[stop_id]] * Td.loc[time_period].iloc[0] * \
                                        beta[beta.stop_id == stop_id].down.iloc[0]
                up = up.append(up_aux)
                down = down.append(down_aux)

        return up, down

    def compute_p(self,tripsloader, usersbystop, service):
        if os.path.exists(RouteTrip.__COEF_BASENAME % service):
            return pd.read_csv(RouteTrip.__COEF_BASENAME % service, index_col = 'stop_id')

        timeseries_o = tripsloader.timeseries_o
        timeseries_d = tripsloader.timeseries_d
        routes = tripsloader.routesLoader.routes.set_index("stop_id")

        stops = routes.loc[usersbystop[usersbystop.stop_id.notna()].stop_id]
        stops = stops[~stops.index.duplicated(keep='first')]

        coef = usersbystop.merge(stops["stop_district"], right_index=True, left_on="stop_id")

        subtimeseries_o = timeseries_o[stops.stop_district.unique()][timeseries_o.index.month == 2]
        subtimeseries_o = subtimeseries_o.between_time("6:00", "0:00")

        subtimeseries_d = timeseries_d[stops.stop_district.unique()][timeseries_d.index.month == 2]
        subtimeseries_d = subtimeseries_d.between_time("6:00", "0:00")

        if service == 'cercanias':
            subtimeseries_d = subtimeseries_d[subtimeseries_d.index.weekday < 5]
            subtimeseries_o = subtimeseries_o[subtimeseries_o.index.weekday < 5]

        subtotal_o = subtimeseries_o.sum()
        subtotal_d = subtimeseries_d.sum()

        subtotal = pd.DataFrame({"o": subtotal_o, "d": subtotal_d})
        coef = coef.merge(subtotal, left_on="stop_district", right_index=True)
        coef["p_o"] = coef["up"] * subtimeseries_o.index.day.unique().shape[0] / coef["o"]
        coef["p_d"] = coef["down"] * subtimeseries_d.index.day.unique().shape[0] / coef["d"]

        if service == 'metro':
            coef['p_o'] = coef['p_o']/29
            coef['p_d'] = coef['p_d']/29
        coef = coef.set_index("stop_id")
        coef.to_csv(RouteTrip.__COEF_BASENAME % service)

        return coef


class DataManager:
    def __init__(self,routetrip):
        self.routetrip = routetrip

        renfe_up,renfe_down = routetrip.get_users_renfe(routetrip.tripsLoader.timeseries_o,
                                                        routetrip.tripsLoader.timeseries_d)

        metro_up,metro_down = routetrip.get_users_metro(routetrip.tripsLoader.timeseries_o,
                                                        routetrip.tripsLoader.timeseries_d,
                                                        max_precision=False)

        self.up = {'cercanias':renfe_up,
                   'metro':metro_up}
        self.down = {'cercanias':renfe_down,
                     'metro':metro_down}
