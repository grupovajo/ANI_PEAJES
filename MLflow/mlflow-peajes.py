#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sabado Nov 11 2023

@author: Equipo VAJO
"""

# Cargue de los datos con sodapy
from sodapy import Socrata
import pandas as pd
import numpy as np
client = Socrata('www.datos.gov.co', None)
tempTr = client.get("8yi9-t44c", limit=200000)
datos = pd.DataFrame.from_records(tempTr)

#Transformaciones
from sklearn.preprocessing import LabelEncoder
datos.desde = pd.to_datetime(datos.desde)
datos.hasta = pd.to_datetime(datos.hasta)
datos.valortarifa = datos.valortarifa.astype('int')
datos.cantidadtrafico = datos.cantidadtrafico.astype('int')
datos.cantidadevasores = datos.cantidadevasores.apply(pd.to_numeric, errors='coerce')
datos.cantidadexentos787 = datos.cantidadexentos787.apply(pd.to_numeric, errors='coerce')
datos['año'] = pd.DatetimeIndex(datos['hasta']).year
datos['mes'] = pd.DatetimeIndex(datos['hasta']).month
datos['añomes'] = pd.to_datetime(datos.hasta).dt.to_period('M')
datos['recaudo'] = datos.valortarifa * datos.cantidadtrafico
datos.idpeaje = datos.idpeaje.astype('int')

cat = LabelEncoder().fit(datos.categoriatarifa)
datos.categoriatarifa = cat.fit_transform(datos.categoriatarifa)

datos.cantidadevasores.fillna(0,inplace=True)
datos.cantidadexentos787.fillna(0,inplace=True)

#Selección de las variables del modelo
columnasX = ['idpeaje','categoriatarifa','año','mes','valortarifa','cantidadevasores','cantidadexentos787']
columnasY = ['cantidadtrafico']

datosT = datos[columnasX+columnasY].groupby(by=['idpeaje', 'categoriatarifa', 'valortarifa', 'año', 'mes'], as_index=False).sum()

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(datosT[columnasX], datosT[columnasY], test_size=0.30, random_state=8)

#Importe de MLFlow para registrar los experimentos, el regresor de XGBoost y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# defina el servidor para llevar el registro de modelos y artefactos
#mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("sklearn-peajes")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    learning_rate = 0.15
    n_estimators=650
    max_depth=10
    min_child_weight=4
    gamma=0
    subsample=0.9
    colsample_bytree=0.8
    nthread=4
    scale_pos_weight=1
    reg_alpha= 0.01
    # Cree el modelo con los parámetros definidos y entrénelo
    xgb = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate,
                               min_child_weight=min_child_weight,gamma=gamma,subsample=subsample,colsample_bytree=colsample_bytree,
                               nthread=nthread,scale_pos_weight=scale_pos_weight,reg_alpha=reg_alpha)
    xgb.fit(Xtrain, ytrain)
    # Realice predicciones de prueba
    predictions = xgb.predict(Xtest)
  
    # Registre los parámetros
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("min_child_weight", min_child_weight)
    mlflow.log_param("reg_alpha", reg_alpha)
  
    # Registre el modelo
    mlflow.sklearn.log_model(xgb, "xgboost-model")
  
    # Cree y registre la métrica de interés
    mse = mean_squared_error(ytest, predictions, squared = False)
    r2 = r2_score(ytest, predictions)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    print(mse)
    print(r2)