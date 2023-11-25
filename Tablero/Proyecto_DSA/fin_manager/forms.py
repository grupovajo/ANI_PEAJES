from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Peajes


peaje=[
    ('ALVARADO', 'ALVARADO'),
    ('ALTAMIRA', 'ALTAMIRA'),
]



class PeajesForm(forms.ModelForm):
    class Meta:
        model = Peajes
        fields =[
                'peaje', 'categoriatarifa', 'valortarifa', 'cantidadtrafico', 'cantidadevasores',
                'cantidadexentos787', 'anio', 'mes', 'aniomes', 'recaudo'
                 ]

        labels = {

            'peaje': 'peaje',
            'categoriatarifa': 'categoriatarifa',
            'valortarifa' : 'valortarifa',
            'cantidadtrafico': 'cantidadtrafico',
            'cantidadevasores': 'cantidadevasores',
            'cantidadexentos787': 'cantidadexentos787',
            'anio': 'anio',
            'mes': 'mes',
            'aniomes': 'aniomes',
            'recaudo': 'recaudo',
            'departamento': 'departamento',
            'municipio': 'municipio',
            'longitud': 'longitud',
            'latitud': 'latitud',

                  }

        widgets = {
            'peaje' : forms.Select(choices=peaje,  attrs={'class': 'mr-sm-2 col-auto my-1 form-row custom-select form-control', 'placeholder' : 'Seleccione el Peaje'}),
            'categoriatarifa': forms.Select(choices=peaje,  attrs={'class': 'mr-sm-2 col-auto my-1 form-row custom-select form-control', 'placeholder' : 'Seleccione'}),
            'valortarifa': forms.NumberInput(attrs={'class': 'form-control'}),
            'cantidadtrafico': forms.NumberInput(attrs={'class': 'form-control'}),
            'cantidadevasores': forms.NumberInput(attrs={'class': 'form-control'}),
            'cantidadexentos787':forms.NumberInput(attrs={'class': 'form-control'}),
            'anio': forms.TextInput(attrs={'class': 'form-control form__input' }),
            'mes': forms.TextInput(attrs={'class': 'form-control form__input' }),
            'aniomes': forms.NumberInput(attrs={'class': 'form-control'}),
            'recaudo': forms.NumberInput(attrs={'class': 'form-control'}),
            'departamento':  forms.TextInput(attrs={'class': 'form-control form__input' }),
            'municipio': forms.TextInput(attrs={'class': 'form-control form__input' }),
            'longitud': forms.NumberInput(attrs={'class': 'form-control'}),
            'latitud': forms.NumberInput(attrs={'class': 'form-control'}),
        }