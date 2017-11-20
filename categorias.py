#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:58:01 2017

@author: ricardo
"""

import pandas 
import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo consolidado
file = pandas.read_csv("consolidado.csv", skiprows = 0, sep = ';', header=0)

# Para obtener el promedio de exito, reemplazar un exito con 1, fracaso con 0
file['state'] = file['state'].replace('successful', 1)
file['state'] = file['state'].replace('failed', 0)

# Por categoría, obtener la cantidad de proyectos,el porcentaje de exitos
# el promedio de patrocinadores
frecuenciaCategoria = file.groupby('category').agg(
        {'backers_count':'mean', 'state':'mean', 'category':'count',
         'goal': 'mean'}
   )
frecuenciaCategoria.columns = ['Patrocinadores', 'Exito', 'Cantidad', 'Objetivo']
frecuenciaCategoria.reset_index(inplace=True)
frecuenciaCategoria.columns = ['Categoria', 'Patrocinadores', 'Exito', 'Cantidad', 'Objetivo']


# reordenar según la cantidad de patrocinadores
frecuenciaCategoria.sort_values(by = ['Patrocinadores'], inplace = True)
frecuenciaCategoria.reset_index(inplace=True, drop = True)

# graficar la cantidad de patrocinadores vs en éxito obtenido
plt.figure()
plt.plot( list(frecuenciaCategoria['Patrocinadores']), 
          list(frecuenciaCategoria['Exito']),
          'ro')
plt.title('Promedio de patrocinadores vs Porcentaje de exito por categoría')
plt.xlabel('Promedio de patrocinadores')
plt.ylabel('Porcentaje de exito')
plt.show(block=False)


# graficar la cantidad de patrocinadores vs en éxito obtenido
plt.figure()
plt.plot( list(frecuenciaCategoria['Objetivo']), 
          list(frecuenciaCategoria['Exito']),
          'ro')
plt.title('Promedio del objetivo vs Porcentaje de exito por categoría')
plt.xlabel('Promedio del objetivo')
plt.ylabel('Porcentaje de exito')
plt.show(block=False)


print('')
print('')
  
print("Correlación entre promedio de patrocinadores y porcentaje de exito entre categorias")
print(np.corrcoef(list(frecuenciaCategoria['Patrocinadores']), 
                    list(frecuenciaCategoria['Exito']))[0][1]
)


print('')
print('')

print(frecuenciaCategoria)
print('')
print('')


frecuenciaCategoria.to_csv("frec.csv", sep = ";", na_rep = '', index = False)

frecuenciaCategoria.sort_values(
        by = ['Patrocinadores'], 
        ascending=[1], 
        inplace = True)
frecuenciaCategoria.reset_index(inplace=True, drop=True)   


print("Categorías más apoyadas: ")
for i in range(0, 5):
    n = len(frecuenciaCategoria)-1-i
    print( frecuenciaCategoria.loc[n, 'Categoria'] + ': ' 
          + str(int(frecuenciaCategoria.loc[n, 'Patrocinadores'])) 
          + ' patrocinadores por proyecto')
 
    
print('')
print('')

print("Categorías menos apoyadas: ")
for i in range(0, 5):
    print( frecuenciaCategoria.loc[i, 'Categoria'] + ': ' 
          + str(int(frecuenciaCategoria.loc[i, 'Patrocinadores']))  
          + ' patrocinadores por proyecto')
    
print('')
print('')   
    
 # reordenar según el exito promedio de la categoría
frecuenciaCategoria.sort_values(
        by = ['Exito'], 
        ascending=[1], 
        inplace = True)
frecuenciaCategoria.reset_index(inplace=True, drop=True)   
 
print('')
print('')
   
    
print("Categorías mas exitosas: ")
for i in range(0, 5):
    n = len(frecuenciaCategoria)-1-i
    print( frecuenciaCategoria.loc[n, 'Categoria'] + ': ' 
          + str(int(frecuenciaCategoria.loc[n, 'Exito'] * 100))  
          + '%')

print('')
print('')


print("Categorías menos exitosas: ")
for i in range(0, 5):
    print( frecuenciaCategoria.loc[i, 'Categoria'] + ': ' 
          + str(int(frecuenciaCategoria.loc[i, 'Exito'] * 100))  
          + '%')
    
  
# reordenar según el objetivo promedio de la categoría
frecuenciaCategoria.sort_values(
        by = ['Objetivo'], 
        ascending=[1], 
        inplace = True)
frecuenciaCategoria.reset_index(inplace=True, drop=True)   
 
print('')
print('')
  
print("Categorías más ambiciosas: ")
for i in range(0, 5):
    n = len(frecuenciaCategoria)-1-i
    print( frecuenciaCategoria.loc[n, 'Categoria'] + ': ' 
          + str(int(frecuenciaCategoria.loc[n, 'Objetivo']))
          + '$ por proyecto')

print('')
print('')


print("Categorías menos ambiciosas: ")
for i in range(0, 5):
    print( frecuenciaCategoria.loc[i, 'Categoria'] + ': ' 
          + str(int(frecuenciaCategoria.loc[i, 'Objetivo']))
          + '$ por proyecto')    
    
    
    
    
    
    
    
    
    
    
    
    
 
print('') 

print("Por categoría, la correlación entre el número de patrocinadores y la probabilidad de exito")
# Ahora por categoría    
categorias = file['category'].unique()
for cat in categorias:
    
    filec = file.loc[file['category'] == cat]
    #filec = file
    filec.reset_index(inplace=True, drop = True)
    
    # organizar por goal
    filec.sort_values(by = ['goal'], inplace = True)
    filec.reset_index(inplace=True, drop = True)
    
    objetivo = []
    exito = []
    
    cantidades = filec['goal'].unique()
    
    for i in cantidades:
        print(i)
        #segmentof = file.loc[i*largo:(i+1)*largo]
        segmentof = filec.loc[filec['backers_count'] == i]
        objetivo.append( int(segmentof['goal'].mean()) )
        exito.append( int(segmentof['state'].mean()*100) )
        
    #print(patrocinadores)
    #print(exito)
    
    #plt.plot(patrocinadores, exito, 'ro')
    
    #print('Categoría ' + cat + ': ' +  str(np.corrcoef(patrocinadores, exito)[1][0]))

    #print('Categoría ' + cat + ': ' +  str(np.corrcoef(patrocinadores, exito)))


#np.corrcoef( list(filec['backers_count']), list(filec['state']) )

#plt.plot( filec.groupby('backers_count')['state'].mean() )

#filec = file.loc[file['category'] == cat]
#plt.plot( list(file['backers_count']), 
#          list(file['state']),
#          'ro')
#plt.show()    
    
































    
    
    
    