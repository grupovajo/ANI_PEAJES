from django.shortcuts import render, redirect
#from django.http import HttpResponse
#from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
#from fin_manager import models
#from django.contrib.auth.decorators import login_required
#from django.db.models import Sum, Count, F
#from django.views.generic import TemplateView
from .models import Account, Liability
from .forms import LiabilityForm
from django.views.generic.edit import FormView
#from django.views.generic import ListView
from django.utils.safestring import mark_safe
#from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
from plotly.graph_objs import *

from sodapy import Socrata
import pandas as pd
from django.contrib.auth.decorators import login_required  # login autenticación

from joblib import load, dump

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



@login_required(redirect_field_name='login')
def home(request):
    return render(request, 'fin_manager/home.html')


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')  # Change 'home' to your desired URL
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})



@login_required(redirect_field_name='login')
def dashboard1(request):
    context={}
    graph_data={}
    graph2_data={}

    # Lectura Fuente Excel
    if "df" in globals():
        print("variable existe")
    else:
        df = pd.read_excel('static/datos/bd_www_datos_gov_co.xlsx', sheet_name='Sheet1', header=0)

    # --------Método POST --------
    if request.method == 'POST':
        Colum1 = request.POST.get('Colum1')
        Colum2 = request.POST.get('Colum2')  # leo la información enviada por medio de POST
        Colum3 = request.POST.get('Colum3')  # leo la información enviada por medio de  POST
        Colum4 = request.POST.get('Colum4')  # leo la información enviada por medio de  POST

        peaje=str(Colum1)
        categoria= str(Colum2)
        anio=str(Colum3)
        mes=str(Colum4)
        mes_anio=anio+'-'+mes

        #Generar gráficas dependiendo  del peaje y mes seleccionado
        # grafica 1
        df1 = df[df['peaje'] == peaje]
        filtered_df = df1[df1['mes'] == mes_anio]
        #print('df2***********************************************')
        #print(df2)
        #filtered_df = df2[df2['categoriatarifa'] == categoria]
        #print('filtered_df***********************************************')
        #print(filtered_df)
        summed_df = filtered_df.groupby(['mes']).sum().reset_index()
        fig = px.line(summed_df, x='mes', y='recaudo', labels={'recaudo': 'Recaudo Total', 'mes': 'Mes'}, title='Recaudo Total por Mes')
        graph_json = fig.to_json()
        graph_data['chart'] = graph_json


        # gráfica 2

        summed_categoria_recaudo = filtered_df.groupby('categoriatarifa')['recaudo'].sum().reset_index()
        fig2 = px.bar(summed_categoria_recaudo, x='categoriatarifa', y='recaudo', color='categoriatarifa',
                      labels={'recaudo': 'Recaudo Total', 'categoriatarifa': 'Categoría de Tarifa'},
                      title='Recaudo Total por Categoría de Tarifa')
        graph2_json = fig2.to_json()
        graph2_data['chart'] = graph2_json



        # Crear un DataFrame para la predicción
        df1 = pd.DataFrame()
        df1['peaje'] = [peaje]
        df1['categoriatarifa'] = [categoria]
        df1['mes'] = [mes_anio]

        # Cargo modelos
        label_encoder_peaje = load('static/modelos/label_encoder_peaje.joblib')
        label_encoder_mes = load('static/modelos/label_encoder_mes.joblib')
        label_encoder_categoria = load('static/modelos/label_encoder_categoria.joblib')
        model = load('static/modelos/RF_regression.joblib')

        # Defino funciones
        def transform_peaje(value):
            return label_encoder_peaje.transform([value])[0]
        def transform_mes(value):
            return label_encoder_mes.transform([value])[0]
        def transform_categoria(value):
            return label_encoder_categoria.transform([value])[0]

        # llamo los objetos para transformar los datos
        df1['peaje'] = df1['peaje'].apply(transform_peaje)
        df1['categoriatarifa'] = df1['categoriatarifa'].apply(transform_categoria)
        df1['mes'] = df1['mes'].apply(transform_mes)


        # Realizar la predicción
        try:
            prediction = model.predict(df1)
            print('prediction  :', prediction)
        except Exception as e:
            print("Error al realizar la predicción:", str(e))
            return f'Error al realizar la predicción: {str(e)}'

        context= { 'Colum1_Select': peaje, 'Colum2_Select' : categoria, 'Colum3_Select': anio, 'Colum4_Select':mes, 'Colum6_Select':prediction[0], 'graph_data': mark_safe(graph_data['chart']), 'graph2_data': mark_safe(graph2_data['chart'])}

        return render(request, 'expenses/dashboard.html', context)

    #grafica 1
    selected_peajes="VALENCIA"
    filtered_df = df[df['peaje']==selected_peajes]
    summed_df = filtered_df.groupby(['mes']).sum().reset_index()
    fig = px.line(summed_df, x='mes', y='recaudo',
                  labels={'recaudo': 'Recaudo Total', 'mes': 'Mes'},
                  title='Recaudo Total por Mes')
    graph_json = fig.to_json()
    graph_data['chart']=  graph_json
    context['graph_data'] = mark_safe(graph_data['chart']) # Use mark_safe to render the JSON as HTML

    #gráfica 2
    selected_meses='2023-01'
    filtered_df = df[df['mes']==selected_meses]
    summed_categoria_recaudo = filtered_df.groupby('categoriatarifa')['recaudo'].sum().reset_index()
    fig2 = px.bar(summed_categoria_recaudo, x='categoriatarifa', y='recaudo', color='categoriatarifa',
                 labels={'recaudo': 'Recaudo Total', 'categoriatarifa': 'Categoría de Tarifa'},
                 title='Recaudo Total por Categoría de Tarifa')
    graph2_json = fig2.to_json()
    graph2_data['chart']=  graph2_json
    context['graph2_data'] = mark_safe(graph2_data['chart']) # Use mark_safe to render the JSON as HTML

    # generar listas de Variables de selección
    context['Colum1_List_views'] = list(df.peaje.unique())
    context['Colum2_List_views'] = list(df.categoriatarifa.unique())
    context['Colum3_List_views'] = [i for i in range(2015, 2023, 1)]
    context['Colum4_List_views'] = ['01','02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    return render(request, 'expenses/dashboard.html', context)

