
# coding: utf-8

# In[1]:

# Importar librerías necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:

# Leer datos

data = pd.read_csv("consolidado.csv", skiprows = 0, sep = ';', header=0)
data.head(100)


# In[3]:

# Calcular la cantidad de proyectos

x = len(data)
x


# In[4]:

# Crear subonjunto de datos solo con US

data_only_US = data.groupby('country')
new_data = data.loc[data_only_US.groups['US']]
new_data.head(30)


# In[5]:

# Calcular la cantidad de proyectos de US

a = len(new_data)
a


# In[6]:

# Crear nuevo dataframe con las columnas goal, pledged, state y category

new_data2 = new_data.loc[:,['goal','pledged','state', 'category']]
new_data2


# In[7]:

# Crear subonjunto de datos del nuevo dataframe para state = successful

new_data3 = new_data2.groupby('state')
new_data_successful = new_data2.loc[new_data3.groups['successful']]
new_data_successful.head(30)


# In[8]:

# Calcular número de proyectos successful

b = len(new_data_successful)
b


# In[9]:

# Crear subonjunto de datos del nuevo dataframe para state = failed

new_data4 = new_data2.groupby('state')
new_data_failed = new_data2.loc[new_data4.groups['failed']]
new_data_failed.head(30)


# In[10]:

# Calcular número de proyectos failed

c = len(new_data_failed)
c


# In[11]:

# Calcular el porcentaje de proyectos successful

percentage_of_successful_projects = b/a
percentage_of_successful_projects


# In[12]:

# Calcular el porcentaje de proyectos failed

percentage_of_failed_projects = c/a
percentage_of_failed_projects


# In[13]:

#Convertir columnas goal y pledged en enteros

new_data_successful['goal'] = pd.to_numeric(new_data_successful['goal']).astype(int)
new_data_successful['pledged'] = pd.to_numeric(new_data_successful['pledged']).astype(int)

new_data_successful


# In[14]:

# Calcular la diferencia entre las columnas goal y pledged

new_data_successful['difference'] = new_data_successful['pledged'] - new_data_successful['goal'] 
new_data_successful.head(30)


# In[15]:

# Extraer difference mayor que cero

d = new_data_successful.loc[new_data_successful['difference'] > 0]
e = len(d)
e


# In[16]:

# Calcular proporción de proyectos con difference mayor que cero 

f = e/b
f


# In[17]:

#Pintar gráfico de proyectos con difference mayor que cero y menor que cero

labels = 'Sobrepasan la meta', 'No sobrepasan la meta'
sizes = [(f),(1-f)]
explode = (0.1, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Proyectos exitosos de Estados Unidos')
plt.show()


# In[18]:

# Ordenar descendentemente

d.sort_values('difference', ascending = False)


# In[19]:

# Dividir en rangos la columna difference y agregar una nueva columna range - Pintar

labels=['[0%-25%]', '(25%-75%]', '(75%-95%]', '(95%-100%]']

d['range']=pd.qcut(d['difference'], [0, .25, .75, .95, 1.],labels=labels)

plt.figure()
bin_diference=d[['range','difference']]
bin_diference=bin_diference.groupby('range').count()
bin_diference.plot.bar(figsize=(10, 6),color='g')
plt.title('Sobrepasos por cuantiles')
plt.xlabel('Cuantiles')
plt.ylabel('Número de proyectos')
plt.show()


# In[20]:

# Calcular ratio de difference sobre goal

d['ratio'] = d['difference'] / d['goal'] 
d.head(30)


# In[23]:

# Pintar muestra de cantidad de veces que se sobrepasan los proyectos

bin_ratio=d[['ratio', 'ratio']]
bin_ratio.columns=['ratio', 'ratio2']

bin_ratio=round(bin_ratio, 0)

bin_ratio=bin_ratio[bin_ratio['ratio']>0]
bin_ratio=bin_ratio[bin_ratio['ratio']<50]

bin_ratio=bin_ratio.groupby('ratio').count()

bin_ratio=bin_ratio[bin_ratio['ratio2']<5000]

frames = [pd.DataFrame(list(bin_ratio.index)), bin_ratio['ratio2']]
bin_ratio4 = pd.concat(frames, axis=1)

bin_ratio4.columns=['ratio', 'ratio2']
bin_ratio4.plot.scatter(x='ratio',y='ratio2', figsize=(10, 6),color='g')

plt.title('Cantidad de veces que se sobrepasan los proyectos')
plt.xlabel('Cantidad')
plt.ylabel('Número de proyectos')
plt.show()

