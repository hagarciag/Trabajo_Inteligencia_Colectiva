#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:48:41 2017

@author: ricardo
"""


import glob
import pandas 

from datetime import datetime

def ReadFile(path):
    file = pandas.read_csv(path, skiprows = 0, sep = ',', header=0,
                 usecols = ['id', 'name', 'blurb', 'goal',
                            'state', 'pledged', 'state', 'country',
                            'currency', 'deadline', 'launched_at',
                            'backers_count', 'creator', 'location',
                            'category', 'spotlight', 'staff_pick'])

    file['location'] = file['location'].str.split('displayable_name').str[1]
    file['location'] = file['location'].str.split('"').str[2]
    
    file['creator'] = file['creator'].str.split(':').str[10]
    file['creator'] = file['creator'].str.split(',').str[0]
    
    file['category'] = file['category'].str.split(':').str[-1]
    file['category'] = file['category'].str.split('"').str[1]
    
    file['subcategory'] = file['category'].str.split('/').str[1]
    file['category'] = file['category'].str.split('/').str[0]
     
    file['deadline']=pandas.to_datetime(file['deadline'], unit='s')
    file['launched_at']=pandas.to_datetime(file['launched_at'], unit='s')
    
    file['blurb'] = file['blurb'].replace(';', '.')
    
    file.drop( 
            file[(file['state'] != 'successful') & (file['state'] != 'failed') ].index, 
            inplace = True)
  
    return file


pathFiles = glob.glob('*.csv')
if ("consolidado.csv" in pathFiles):
    pathFiles.remove('consolidado.csv')
bigFile = pandas.DataFrame()
    
# Procesa cada archivo y lo anexa a un solo DataFrame a regresar
for path in pathFiles:
    print(path)
    bigFile = bigFile.append(ReadFile(path), ignore_index = True)

# eliminar filas duplicadas
bigFile.drop_duplicates(inplace = True)


# Guardar archivo
print("Guardando archivo consolidado")
bigFile.to_csv("consolidado.csv", sep = ";", na_rep = '', index = False)




#linea = bigFile['location']


#import pandas 
#file = pandas.read_csv("consolidado.csv", skiprows = 0, sep = ';', header=0)
























