#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:48:41 2017

@author: ricardo
"""

# Paquetes a importar, para manejar los datos en un dataframe
import glob
import pandas
# Paquetes para descargar el archivo de la web y descomprimirlo
import requests, zipfile, io


def ReadFile(path):
    # Dada la ruta de un archivo, se lee este, y se separan ciertas columnas de interés para analizar
    # no todas las columnas tienen información relevante
    file = pandas.read_csv(path, skiprows = 0, sep = ',', header=0,
                 usecols = ['id', 'name', 'blurb', 'goal',
                            'state', 'pledged', 'state', 'country',
                            'currency', 'deadline', 'launched_at',
                            'backers_count', 'creator', 'location',
                            'category', 'spotlight', 'staff_pick'])

    # se filtran los proyectos por aquellos que fueron exitoso o fallidos
    # pues el objetivo es predecir si un poryecto será exitoso o no
    file.drop( 
            file[(file['state'] != 'successful') & (file['state'] != 'failed') ].index, 
            inplace = True)
    
    # Dado que mas del 80% de los proyectos son de los Estados Unidos, y para evitar problemas de conversión
    # entre monedas, se opta por solo operar con este país
    file.drop( 
            file[(file['country'] != 'US')].index, 
            inplace = True)
    
    # La columna location tiene un formato json con muchos campos innecesarios
    # se debe separar la información pedida, en este caso la ciudad en donde se realizó el proyecto
    file['location'] = file['location'].str.split('displayable_name').str[1]
    file['location'] = file['location'].str.split('"').str[2]
    
    # La columna creator tiene un formato json con muchos campos innecesarios
    # se debe separar la información pedida, en este caso el nombre de la persona o entidad que creó el proyecto
    file['creator'] = file['creator'].str.split(':').str[10]
    file['creator'] = file['creator'].str.split(',').str[0]
    
    # la columna category tambien tiene un formato json con campos innecesarios, se separa la categoría del proyecto
    file['category'] = file['category'].str.split(':').str[-1]
    file['category'] = file['category'].str.split('"').str[1]
    # Tambien se obtiene la subcategoría de cada proyecto
    file['subcategory'] = file['category'].str.split('/').str[1]
    file['category'] = file['category'].str.split('/').str[0]
    
    # Los campos deadline (que indica el momento en que terminó el proyecto) y 
    # launched_at (que indica el momento en que inició un proyecto) están en formato timestamp de unix
    # así que se deben transformar a una fecha legible
    file['deadline']=pandas.to_datetime(file['deadline'], unit='s')
    file['launced_at']=pandas.to_datetime(file['launched_at'], unit='s')
    
    # Este campo tiene la descripción del proyecto, para evitar problemas de lectura posteriores
    # se reemplazan los ; por .
    file['blurb'] = file['blurb'].replace(';', '.')
  
    file.reset_index(inplace=True, drop = True)
    # Agregar la duración del proyecto en días en la plataforma, para analizar tiempos de duración
    # de un proyecto
    file['time'] = 0
    for i in range(0, len(file)):
        inicio = file.loc[i, 'launched_at']
        fin = file.loc[i, 'deadline']
        
        file.loc[i, 'time'] = (fin - inicio).days

    return file

# Se descarga el archivo .zip de la web del autor y se descomprime en la carpeta actual
# Descarga y descompresion de los archivos
'''
url = "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2017-10-15T10_20_38_271Z.zip"
r = requests.get(url, stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
'''


# Se toma el nombre de todos los archivos .csv que existan en la carpeta actual
pathFiles = glob.glob('*.csv')
# se descarta el archivo consolidado.csv, que será el archivo con todos los datos en uno solo
if ("consolidado.csv" in pathFiles):
    pathFiles.remove('consolidado.csv')
    
# este será el dataframe que contendrá la información
bigFile = pandas.DataFrame()
    
# Procesa cada archivo y lo anexa a un solo DataFrame
for path in pathFiles:
    print(path)
    bigFile = bigFile.append(ReadFile(path), ignore_index = True)

# eliminar filas duplicadas
bigFile.drop_duplicates(inplace = True)


# Guardar archivo
print("Guardando archivo consolidado")
bigFile.to_csv("consolidado.csv", sep = ";", na_rep = '', index = False)
























