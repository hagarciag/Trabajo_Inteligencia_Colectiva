
# coding: utf-8

# In[147]:

# Importar librerías necesarias

import pandas as pd
import numpy as np


# In[148]:

# Leer datos

file = pd.read_csv("consolidado.csv", skiprows = 0, sep = ';', header=0)


# In[149]:

# Calcular la cantidad de proyectos

a = len(file)
a


# In[150]:

# Crear nuevo dataframe con las columnas goal, pledged y state

new_file = file.loc[:,['goal','pledged','state']]
new_file.head(30)


# In[151]:

# Crear subonjunto de datos del nuevo dataframe para state = successful

new_file_state = new_file.groupby('state')
new_file_state_successful = new_file.loc[new_file_state.groups['successful']]
new_file_state_successful.head(30)


# In[152]:

# Calcular número de proyectos successful

#b = new_file_state_successful['state'].value_counts()
len(new_file_state_successful)


# In[153]:

# Calcular el porcentaje de proyectos successful

percentage_of_successful_projects = b/a
percentage_of_successful_projects


# In[154]:

# Calcular la diferencia entre las columnas goal y pledged

new_file_state_successful['difference'] = new_file_state_successful['pledged'] - new_file_state_successful['goal'] 
new_file_state_successful.head(30)


# In[155]:

# Ordenar descendentemente

new_file_state_successful.sort_values('difference', ascending = False)


# In[156]:

# Extraer difference mayor que cero

c = new_file_state_successful.loc[new_file_state_successful['difference'] > 0]
len(c)


# In[157]:

# Calcular el número y porcentaje de proyectos successful con difference mayor que cero 

projects_number_0 = len(new_file_state_successful) - len(c)
percentage_of_successful_projects_0 = len(c) / len(new_file_state_successful)
print(projects_number_0)
print(percentage_of_successful_projects_0)


# In[158]:

# Ordenar descendentemente difference mayor que cero 

d = c.sort_values('difference', ascending = False)
d


# In[159]:

# Calcular ratio de difference sobre goal

d['ratio'] = d['difference'] / d['goal'] 
d.head(30)


# In[160]:

d['goal'] = pd.to_numeric(d['goal']).astype(float)
d['pledged'] = pd.to_numeric(d['pledged']).astype(float)
d['difference'] = pd.to_numeric(d['difference']).astype(float)
d


# In[161]:

d['ratio'] = d['difference'] / d['goal'] 
d.head(30)


# In[ ]:



