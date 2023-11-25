#from django.conf.urls import url
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard1', views.dashboard1, name='dashboard1'),
    path('accounts/register/', views.register, name='register'),
    path('cargar_datos', views.cargar_datos, name='cargar_datos'),

]
