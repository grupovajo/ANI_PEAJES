from django.shortcuts import render, redirect
from django.http import HttpResponse
#from django.shortcuts import HttpResponse
#from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
#from fin_manager import models
#from django.contrib.auth.decorators import login_required
#from django.views.generic import TemplateView
from .models import Peajes
from django.views.generic.edit import FormView
#from django.views.generic import ListView
from django.utils.safestring import mark_safe
from dateutil.relativedelta import relativedelta
import plotly.express as px
from plotly.graph_objs import *

from sodapy import Socrata
import pandas as pd
from django.contrib.auth.decorators import login_required  # login autenticación

from joblib import load, dump
from datetime import datetime
from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import json
import numpy as np
#from django.http import is_ajax






def home(request):
    #return render(request, 'fin_manager/home.html')
    return redirect('/dashboard1')

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



def dashboard1(request):


    # Actualizo la información por medio de Ajax
    if "X-Requested-With" in request.headers and request.headers["X-Requested-With"] == "XMLHttpRequest":
        print('por dentro de ajax-----------------------------------------------')
        Colum10 = request.GET.get('Colum10')
        Colum20 = request.GET.get('Colum20')  # leo la información enviada por medio de Ajax
        Colum30 = request.GET.get('Colum30')  # leo la información enviada por medio de Ajax
        Colum40 = request.GET.get('Colum40')  # leo la información enviada por medio de Ajax
        Colum50 = request.GET.get('Colum50')  # leo la información enviada por medio de Ajax



        if Colum40  is not None and Colum50 is not None:
            print('por dentro de ajax 1-----------------------------------------------')
            df1_Colum5  = pd.DataFrame(list(Peajes.objects.filter(anio = Colum10).filter(mes = Colum20).filter(departamento = Colum30).filter(municipio = Colum40).filter(categoriatarifa = Colum50).values()))
            list_Colum5 = list(df1_Colum5['peaje'])
            list_Colum5  = list(np.unique(list_Colum5))
            data = json.dumps(list_Colum5 )
            return HttpResponse(data, 'application/json')

        if Colum40  is not None and Colum50 is None:
            print('por dentro de ajax 1-----------------------------------------------')
            df1_Colum4  = pd.DataFrame(list(Peajes.objects.filter(anio = Colum10).filter(mes = Colum20).filter(departamento = Colum30).filter(municipio = Colum40).values()))
            list_Colum4 = list(df1_Colum4['categoriatarifa'])
            list_Colum4  = list(np.unique(list_Colum4 ))
            print('List_Colum4 ,---------------------------------------------------------', list_Colum4)
            data = json.dumps(list_Colum4 )
            return HttpResponse(data, 'application/json')

        if Colum30 is not None and Colum40 is None:
            print('por dentro de ajax2-----------------------------------------------')
            df1_Colum3  = pd.DataFrame(list(Peajes.objects.filter(anio = Colum10).filter(mes = Colum20).filter(departamento = Colum30).values()))
            list_Colum3 = list(df1_Colum3['municipio'])
            list_Colum3 = list(np.unique(list_Colum3))
            print('List_Colum3 ,---------------------------------------------------------', list_Colum3)
            data = json.dumps(list_Colum3)
            return HttpResponse(data, 'application/json')

        if Colum20 is not None and  Colum30 is None:
            print('por dentro de ajax 3-----------------------------------------------')
            df1_Colum2  = pd.DataFrame(list(Peajes.objects.filter(anio = Colum10).filter(mes = Colum20).values()))
            List_Colum2 = list(df1_Colum2['departamento'])
            List_Colum2 =list(np.unique(List_Colum2))
            print('List_Colum2 ,---------------------------------------------------------', List_Colum2 )
            data = json.dumps(List_Colum2 )
            return HttpResponse(data, 'application/json')

        if Colum10 is not None and  Colum20 is None:
            print('por dentro de ajax 4-----------------------------------------------')
            df1_Colum1  = pd.DataFrame(list(Peajes.objects.filter(anio = Colum10).values()))
            list_colum1= list(df1_Colum1['mes'])
            list_colum1= list(np.unique(list_colum1))
            print('list_colum1,---------------------------------------------------------', list_colum1)
            data = json.dumps(list_colum1)
            return HttpResponse(data, 'application/json')






    context={}

    #df = pd.DataFrame(list(Peajes.objects.filter(anio=2015).values()))
    df = pd.DataFrame(list(Peajes.objects.values()))
    # Unión de idpeaje y peaje
    df['idpeaje_peaje']= df['idpeaje'].astype(str) + ". " +df['peaje']


    # generar listas de Variables de selección modelo
    context['Colum1_List_views'] = list(df.idpeaje_peaje.unique())
    df.categoriatarifa=df.categoriatarifa.astype(int)
    Colum2_List_views= list(df.categoriatarifa.sort_values().unique())
    context['Colum2_List_views']=Colum2_List_views



    # --------Método POST --------
    if request.method == 'POST':
        try:
            # listas de selección de filtros
            context['Colum1_List_views0'] = list(df.anio.unique())
            context['Colum2_List_views0'] = list(df.mes.unique())
            context['Colum3_List_views0'] = list(df.departamento.unique())
            context['Colum4_List_views0'] = list(df.municipio.unique())
            context['Colum5_List_views0'] = list(df.categoriatarifa.unique())  # Categoria
            context['Colum6_List_views0'] = list(df.peaje.unique())  # peaje

            Colum0 = request.POST.get('Colum0')
            Colum1 = request.POST.get('Colum1')
            Colum2 = request.POST.get('Colum2')  # leo la información enviada por medio de POST
            Colum3 = request.POST.get('Colum3')  # leo la información enviada por medio de  POST
            Colum4 = request.POST.get('Colum4')  # leo la información enviada por medio de  POST
            Colum5 = request.POST.get('Colum5')  # leo la información enviada por medio de  POST
            try:
                cantidad_exentos = float(Colum0.replace(',', ''))
            except:
                cantidad_exentos = 0

            if Colum0 is not None or Colum1 is not None or Colum2 is not None or Colum3 is not None or Colum4 is not None or Colum5 is not None:
                peaje = str(Colum1)
                categoria = int(Colum2)
                fecha = datetime.fromisoformat(Colum3)
                valor_tarifa = float(Colum4.replace(',', ''))
                cantidad_evasores = float(Colum5.replace(',', ''))

                # Crear un DataFrame para la predicción # Se realizó de acuerdo a como se entrenó el modelo# (año)
                df1 = pd.DataFrame()
                df1['idpeaje'] = [int(peaje.split('.')[0])]
                df1['categoriatarifa'] = [categoria]
                df1['año'] = [int(fecha.year)]
                df1['mes'] = [int(fecha.month)]
                df1['valortarifa'] = [valor_tarifa]
                df1['cantidadevasores'] = [cantidad_evasores]
                df1['cantidadexentos787'] = [cantidad_exentos]

                # Cargo modelos
                model = load('static/modelos/modelo_xgboost_peajes.joblib')

                # Realizar la predicción
                prediction = model.predict(df1)

                # gráficas
                graph_data = mark_safe(grafica_1(df))
                graph2_data = mark_safe(grafica_2(df))
                graph3_data = mark_safe(grafica_3(df))

                # 'Colum5_Select': Colum5
                context2 = {'graph_data': graph_data, 'graph2_data': graph2_data,
                           'graph3_data': graph3_data, 'Colum0_Select': Colum0, 'Colum1_Select': Colum1,
                           'Colum2_Select': Colum2, 'Colum3_Select': Colum3, 'Colum4_Select': Colum4,
                           'Colum6_Select': prediction[0]}
                context3 = context2 | context

                return render(request, 'expenses/dashboard.html', context3)

            # Opción 2 método POST () filtros para las gráficas----------------

            Colum10 = request.POST.get('Colum10', None)
            Colum20 = request.POST.get('Colum20', None)  # leo la información enviada por medio de Ajax
            Colum30 = request.POST.get('Colum30', None)  # leo la información enviada por medio de Ajax
            Colum40 = request.POST.get('Colum40', None)  # leo la información enviada por medio de Ajax
            Colum50 = request.POST.get('Colum50', None)  # leo la información enviada por medio de Ajax
            Colum60 = request.POST.get('Colum60', None)  # leo la información enviada por medio de Ajax

            if Colum10 != 'Todos' or Colum20 != 'Todos' or Colum30 != 'Todos' or Colum40 != 'Todos' or Colum50 != 'Todos' or Colum60 != 'Todos':

                if Colum10 != 'Todos' and Colum10 is not None and Colum10 != '':
                    df = df[df['anio'] == int(Colum10)]
                if Colum20 != 'Todos' and Colum20 is not None and Colum20 != '':
                    df = df[df['mes'] == int(Colum20)]
                if Colum30 != 'Todos' and Colum30 is not None and Colum30 != '':
                    df = df[df['departamento'] == Colum30]
                if Colum40 != 'Todos' and Colum40 is not None and Colum40 != '':
                    df = df[df['municipio'] == Colum40]
                if Colum50 != 'Todos' and Colum50 is not None and Colum50 != '':
                    df = df[df['categoriatarifa'] == float(Colum50)]
                if Colum60 != 'Todos' and Colum60 is not None and Colum60 != '':
                    df = df[df['peaje'] == (Colum60)]

                # Valores de filtros
                context['Colum1_Select0'] = Colum10
                context['Colum2_Select0'] = Colum20
                context['Colum3_Select0'] = Colum30
                context['Colum4_Select0'] = Colum40
                context['Colum5_Select0'] = Colum50
                context['Colum6_Select0'] = Colum60

                # Unión de idpeaje y peaje
                df['idpeaje_peaje'] = df['idpeaje'].astype(str) + ". " + df['peaje']

                # gráficas
                context['graph_data'] = mark_safe(grafica_1(df))
                context['graph2_data'] = mark_safe(grafica_2(df))
                context['graph3_data'] = mark_safe(grafica_3(df))
            else:
                return redirect('/dashboard1')  # Change  your desired URL

            return render(request, 'expenses/dashboard.html', context)
        except:
            return redirect('/dashboard1')  # Change  your desired URL


    #gráficas
    context['graph_data'] = mark_safe(grafica_1(df))
    context['graph2_data'] = mark_safe(grafica_2(df))
    context['graph3_data'] = mark_safe(grafica_3(df))


    # Valores de filtros
    context['Colum1_List_views0'] = list(df.anio.unique())
    context['Colum2_List_views0'] = list(df.mes.unique())
    context['Colum3_List_views0'] = list(df.departamento.unique())
    context['Colum4_List_views0'] = list(df.municipio.unique())
    context['Colum5_List_views0'] = Colum2_List_views # Categoria
    context['Colum6_List_views0'] = list(df.peaje.unique()) #peaje

    return render(request, 'expenses/dashboard.html', context)



def grafica_1(filtered_df):
    graph_data={}
    # grafica 1
    summed_df = filtered_df.loc[:, ['idpeaje', 'categoriatarifa', 'anio', 'mes', 'valortarifa', 'cantidadevasores', 'cantidadexentos787', 'cantidadtrafico', 'recaudo']].groupby(['mes']).sum().reset_index()
    fig = px.line(summed_df, x='mes', y='recaudo', labels={'recaudo': 'Recaudo Total', 'mes': 'Mes'},  title='Recaudo Total por Mes')
    graph_json = fig.to_json()
    graph_data['chart'] = graph_json
    return graph_data['chart']

def grafica_2(filtered_df):
    # gráfica 2
    graph2_data={}
    summed_categoria_recaudo = filtered_df.loc[:, ['idpeaje', 'categoriatarifa', 'anio', 'mes', 'valortarifa', 'cantidadevasores','cantidadexentos787', 'cantidadtrafico', 'recaudo']].groupby('categoriatarifa')['recaudo'].sum().reset_index()
    fig2 = px.bar(summed_categoria_recaudo, x='categoriatarifa', y='recaudo', color='categoriatarifa',labels={'recaudo': 'Recaudo Total', 'categoriatarifa': 'Categoría de Tarifa'}, title='Recaudo Total por Categoría de Tarifa')
    graph2_json = fig2.to_json()
    graph2_data['chart'] = graph2_json
    return graph2_data['chart']

def grafica_3(filtered_df):
    # gráfica 3
    graph3_data = {}
    # Quito los valores negativos
    filtered_df['recaudo'] = filtered_df['recaudo'].where(filtered_df['recaudo'] >= 0, 0)
    fig3 = px.scatter_mapbox(filtered_df, lat="latitud", lon="longitud", hover_name="peaje",hover_data=["departamento", "municipio"], color='recaudo', size='recaudo',color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=4, height=500)
    fig3.update_layout(mapbox_style="open-street-map")
    fig3.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    graph3_json = fig3.to_json()
    graph3_data['chart'] = graph3_json
    return graph3_data['chart']




#---------------------------Cargar datos en el MODELO--------------------------------------
def cargar_datos(request):
    archivo = 'static/datos/bd_www_datos_gov_co.xlsx'
    df1 = pd.read_excel(archivo, sheet_name="Sheet1", header=0, na_values="-")
    '''
    # Determino las tablas de información que se van a leer
    df2 = df1.fillna(0)  # reemplazo los valores "nan" por cero "0"

    # Elimino la información inicial que se encuentra en el modelo para cargar los nuevos datos
    datos_modelo_cargados = Peajes.objects.all()
    try:
        datos_modelo_cargados.delete()
    except:
        print('No hay datos en la tabla peajes')

    # Intento de llenar la tabla
    for fila in range(df2.shape[0]):
        datos_modelo = Peajes(
            id=int(df2.iloc[fila, 0]),
            idpeaje=int(df2.iloc[fila, 1]),
            peaje=str(df2.iloc[fila, 2]),
            categoriatarifa=float(df2.iloc[fila, 3]),
            desde=str(df2.iloc[fila, 4]),
            hasta=str(df2.iloc[fila, 5]),
            valortarifa=float(df2.iloc[fila, 6]),
            cantidadtrafico=float(df2.iloc[fila, 7]),
            cantidadevasores=float(df2.iloc[fila, 8]),
            cantidadexentos787=float(df2.iloc[fila, 9]),
            anio=int(df2.iloc[fila, 10]),
            mes=int(df2.iloc[fila, 11]),
            aniomes=str(df2.iloc[fila, 12]),
            recaudo=float(df2.iloc[fila, 13]),
            departamento=str(df2.iloc[fila, 14]),
            municipio = str(df2.iloc[fila, 15]),
            longitud = float(df2.iloc[fila, 16]),
            latitud = float(df2.iloc[fila, 17]),
        )
        datos_modelo.save()
    '''
    return redirect('/dashboard1')
