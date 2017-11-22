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
labels=['[1-2]', '(2-6]', '(6-14]', '(14-29]', '(29-51]', '(51-88]', '(88-189]', '(189~106K]']
data_bakers=data.loc[data['backers_count'] > 0]
data=data[data['country']=='US']
#print(data_bakers.columns)

data_bakers['Cant Patrocinadores']=pd.qcut(data_bakers['backers_count'], 8,labels=labels)

#Estados################################################################################
df_state = pd.get_dummies(data_bakers['state'])
data_bakers = data_bakers.join(df_state)

bins_states=data_bakers[['Cant Patrocinadores', 'successful', 'failed']]

bins_states.columns=['Cant Patrocinadores', 'Cantidad Campañas Exitosas', 'Cantidad Campañas Fallidas']

bins_states=bins_states.groupby('Cant Patrocinadores').sum()




#Objetivos vs patrocinadores###########################################################

labels=['(0-750]', '(750-1700]', '(1700-3K]', '(3K-5K]', '(5K-8K]', '(8K-15K]', '(15K-30K]', '(30K-100M]']
data_objetivo=data.loc[data['goal'] > 0]

df_state = pd.get_dummies(data_bakers['state'])
data_objetivo = data_objetivo.join(df_state)

data_objetivo['failed_bakers']=data_objetivo['failed']*data_objetivo['backers_count']
data_objetivo['successful_bakers']=data_objetivo['successful']*data_objetivo['backers_count']

data_objetivo['Meta']=pd.qcut(data_objetivo['goal'], 8,labels=labels)

bins_objetivo_bakers=data_objetivo[['Meta', 'successful_bakers', 'failed_bakers']]

bins_objetivo_bakers=bins_objetivo_bakers.groupby('Meta').sum()

bins_objetivo_bakers_porc_falledo=(bins_objetivo_bakers['failed_bakers']/(bins_objetivo_bakers['failed_bakers']+bins_objetivo_bakers['successful_bakers']))*100
bins_objetivo_bakers_porc_exito=(bins_objetivo_bakers['successful_bakers']/(bins_objetivo_bakers['failed_bakers']+bins_objetivo_bakers['successful_bakers']))*100
bins_objetivo_bakers['successful_bakers']=bins_objetivo_bakers_porc_exito
bins_objetivo_bakers['failed_bakers']=bins_objetivo_bakers_porc_falledo

bins_objetivo_bakers.columns=['% Patrocinadores Campañas Exitosas', '% Patrocinadores Campañas Fallidas']




#Objetivos vs recolectado######################################################

data_objetivo['failed_recogido']=(data_objetivo['failed']*data_objetivo['goal'])/1000000
data_objetivo['successful_recogido']=(data_objetivo['successful']*data_objetivo['goal'])/1000000

bins_objetivo_recogido=data_objetivo[['Meta', 'successful_recogido', 'failed_recogido']]

bins_objetivo_recogido.columns=['Meta', 'Millones Recolectado Campañas Exitosas', 'Millones Recolectado Campañas Fallidas']

bins_objetivo_recogido=bins_objetivo_recogido.groupby('Meta').sum()




#Objetivos vs exito###########################################################

data_objetivo['failed_recogido']=(data_objetivo['failed']*data_objetivo['goal'])/1000000
data_objetivo['successful_recogido']=(data_objetivo['successful']*data_objetivo['goal'])/1000000

bins_objetivo_exito=data_objetivo[['Meta', 'successful_recogido', 'failed_recogido']]

bins_objetivo_exito.columns=['Meta', 'Exitoso', 'Fallido']

bins_objetivo_exito=bins_objetivo_exito.groupby('Meta').sum()

bins_objetivo_exito_porc_falledo=(bins_objetivo_exito['Fallido']/(bins_objetivo_exito['Fallido']+bins_objetivo_exito['Exitoso']))*100
bins_objetivo_exito_porc_exito=(bins_objetivo_exito['Exitoso']/(bins_objetivo_exito['Fallido']+bins_objetivo_exito['Exitoso']))*100
bins_objetivo_exito['Exitoso']=bins_objetivo_exito_porc_exito
bins_objetivo_exito['Fallido']=bins_objetivo_exito_porc_falledo

bins_objetivo_exito.columns=['% Exitos', '% Fallos']



#Objetivo - Prop objetivo/Bakers########################################################################
bins_prop_meta_patr=data_objetivo[['Meta', 'goal', 'backers_count']]

bins_prop_meta_patr.columns=['Meta', 'Objetivo', 'Patrocinadores']

bins_prop_meta_patr=bins_prop_meta_patr.groupby('Meta').sum()

bins_prop_meta_patr['Patrocinadores/Meta']=bins_prop_meta_patr['Patrocinadores']/bins_prop_meta_patr['Objetivo']

bins_prop_meta_patr=bins_prop_meta_patr[['Patrocinadores/Meta']]

print(bins_prop_meta_patr.columns)
bins_prop_meta_patr.columns=['Patrocinadores/Meta']

#######################################################################################################

bins_prop_meta_patr.plot.bar(figsize=(10, 6));
#El exito de un proyecto depende del volumen de apoyo que recibo.
bins_states.plot.bar(figsize=(10, 6));
# que apoyan las campañas exitosas. Es decir, muchos patrocinadores apoyan unos pocos proyectos
bins_objetivo_bakers.plot.bar(figsize=(10, 6));
#La cantidad de dinero recolectado aumenta con la meta. Entonces no se ve un fenomeno marcado de pocos dan mucho.
bins_objetivo_recogido.plot.bar(figsize=(10, 6));
#Mientras mayor sea la meta mas probabilidad de que falle el proyecto
bins_objetivo_exito.plot.bar(figsize=(10, 6));

plt.show()

