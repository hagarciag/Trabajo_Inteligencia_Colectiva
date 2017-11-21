
# coding: utf-8

# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Lectura archivo
data=pd.read_csv('consolidado.csv',  # el archivo
                    sep = ';',         # separador de campos
                    thousands = None,  # separador de miles para números
                    decimal = '.',
                    skiprows = 0,
                    header=0)     # separador de los decimales para números
labels=['[1-2]', '(2-6]', '(6-14]', '(14-29]', '(29-51]', '(51-88]', '(88-189]', '(189-105857]']
data_bakers=data.loc[data['backers_count'] > 0]
#print(data_bakers.columns)

data_bakers['Cant Patrocinadores']=pd.qcut(data_bakers['backers_count'], 8,labels=labels)

#Dinero obtenido
bins_dinero=data_bakers[['Cant Patrocinadores', 'goal', 'pledged']]

bins_dinero.columns=['Cant Patrocinadores', 'Objetivo', 'Obtenido']

bins_dinero=bins_dinero.groupby('Cant Patrocinadores').sum()

#Estados
df_state = pd.get_dummies(data_bakers['state'])
data_bakers = data_bakers.join(df_state)

bins_states=data_bakers[['Cant Patrocinadores', 'failed', 'successful']]

bins_states.columns=['Cant Patrocinadores', 'Fallido', 'Exitoso']

bins_states=bins_states.groupby('Cant Patrocinadores').sum()

#Objetivos vs patrocinadores

labels=['(0-750]', '(750-1700]', '(1700-3000]', '(3000-5000]', '(5000-8000]', '(8000-15000]', '(15000-30000]', '(30000-100 Mill]']
data_objetivo=data.loc[data['goal'] > 0]


#Objetivo vs Patrocinadores
df_state = pd.get_dummies(data_bakers['state'])
data_objetivo = data_objetivo.join(df_state)

data_objetivo['failed_bakers']=data_objetivo['failed']*data_objetivo['backers_count']
data_objetivo['successful_bakers']=data_objetivo['successful']*data_objetivo['backers_count']

data_objetivo['Valor Objetivo']=pd.qcut(data_objetivo['goal'], 8,labels=labels)

bins_objetivo_bakers=data_objetivo[['Valor Objetivo', 'failed_bakers', 'successful_bakers']]

bins_objetivo_bakers=bins_objetivo_bakers.groupby('Valor Objetivo').sum()

#print(bins_objetivo_bakers)
bins_objetivo_bakers_porc_falledo=(bins_objetivo_bakers['failed_bakers']/(bins_objetivo_bakers['failed_bakers']+bins_objetivo_bakers['successful_bakers']))*100
bins_objetivo_bakers_porc_exito=(bins_objetivo_bakers['successful_bakers']/(bins_objetivo_bakers['failed_bakers']+bins_objetivo_bakers['successful_bakers']))*100
bins_objetivo_bakers['successful_bakers']=bins_objetivo_bakers_porc_exito
bins_objetivo_bakers['failed_bakers']=bins_objetivo_bakers_porc_falledo

#print(bins_objetivo_bakers)

bins_objetivo_bakers.columns=['% Patrocinadores Campañas Fallidas', '% Patrocinadores Campañas Exitosas']



#Objetivos vs recolectado

data_objetivo['failed_recogido']=(data_objetivo['failed']*data_objetivo['pledged'])/1000000
data_objetivo['successful_recogido']=(data_objetivo['successful']*data_objetivo['pledged'])/1000000

bins_objetivo_recogido=data_objetivo[['Valor Objetivo', 'failed_recogido', 'successful_recogido']]

bins_objetivo_recogido.columns=['Valor Objetivo', 'Millones Recolectado Campañas Fallidas', 'Millones Recolectado Campañas Exitosas']

bins_objetivo_recogido=bins_objetivo_recogido.groupby('Valor Objetivo').sum()



#Objetivos vs exito

data_objetivo['failed_recogido']=(data_objetivo['failed']*data_objetivo['pledged'])/1000000
data_objetivo['successful_recogido']=(data_objetivo['successful']*data_objetivo['pledged'])/1000000

bins_objetivo_exito=data_objetivo[['Valor Objetivo', 'failed', 'successful']]

bins_objetivo_exito.columns=['Valor Objetivo', 'Fallido', 'Exitoso']

bins_objetivo_exito=bins_objetivo_exito.groupby('Valor Objetivo').sum()

bins_objetivo_exito_porc_falledo=(bins_objetivo_exito['Fallido']/(bins_objetivo_exito['Fallido']+bins_objetivo_exito['Exitoso']))*100
bins_objetivo_exito_porc_exito=(bins_objetivo_exito['Exitoso']/(bins_objetivo_exito['Fallido']+bins_objetivo_exito['Exitoso']))*100
bins_objetivo_exito['Exitoso']=bins_objetivo_exito_porc_exito
bins_objetivo_exito['Fallido']=bins_objetivo_exito_porc_falledo

bins_objetivo_exito.columns=['% Fallos', '% Exitos']

#El exito de un proyecto depende del volumen de apoyo que recibo.
bins_states.plot.bar(stacked=True,figsize=(10, 6));
#bins_dinero.plot.bar(figsize=(10, 6));
#La cantidad de dinero recolectado aumenta con la meta. Entonces no se ve un fenomeno marcado de pocos dan mucho.
bins_objetivo_recogido.plot.bar(stacked=True,figsize=(10, 6));
#Mientras mayor sea la meta mas probabilidad de que falle el proyecto
bins_objetivo_exito.plot.bar(figsize=(10, 6));
#A pesar de que la probabilidad de fallo aumenta con el valor de  la meta, no cambia el porcentaje de patrocinadores 
# que apoyan las campañas exitosas. Es decir, muchos patrocinadores apoyan unos pocos proyectos
bins_objetivo_bakers.plot.bar(figsize=(10, 6));

plt.show()


