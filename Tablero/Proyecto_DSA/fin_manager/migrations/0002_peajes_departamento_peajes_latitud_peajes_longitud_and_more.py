# Generated by Django 4.2.7 on 2023-11-22 02:44

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fin_manager", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="peajes",
            name="departamento",
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
        migrations.AddField(
            model_name="peajes",
            name="latitud",
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name="peajes",
            name="longitud",
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name="peajes",
            name="municipio",
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]
