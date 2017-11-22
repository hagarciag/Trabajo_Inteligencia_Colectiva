###Pronostico de exito o fallo de campañas en estado live
import glob
import pandas
from sklearn import tree  
import pydotplus  
from IPython.display import Image  
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import sklearn.metrics as sm
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier


def ReadFile(path):
    # Dada la ruta de un archivo, se lee este, y se separan ciertas columnas de interés para analizar
    # no todas las columnas tienen información relevante
    file = pandas.read_csv(path, skiprows = 0, sep = ',', header=0,
                 usecols = ['id', 'name', 'blurb', 'goal',
                             'pledged', 'state', 'country',
                            'currency', 'deadline', 'launched_at',
                            'backers_count', 'creator', 'location',
                            'category', 'spotlight', 'staff_pick'])

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
    file['launched_at']=pandas.to_datetime(file['launched_at'], unit='s')
    
    # Este campo tiene la descripción del proyecto, para evitar problemas de lectura posteriores
    # se reemplazan los ; por .
    file['blurb'] = file['blurb'].replace(';', '.')
    
    # se dejan solo las campañas que aun estan vivas 
    # pues el objetivo es predecir si al final será exitosa
    file.drop( 
            file[(file['state'] != 'live')].index, 
            inplace = True)
  
    return file

# Se toma el nombre de todos los archivos .csv que existan en la carpeta actual
pathFiles = glob.glob('*.csv')
# se descarta el archivo consolidado.csv, que será el archivo con todos los datos en uno solo
if ("consolidado.csv" in pathFiles):
    pathFiles.remove('consolidado.csv')
    
if ("dataset_arbol.csv" in pathFiles):
    pathFiles.remove('dataset_arbol.csv')

if ("fallos.csv" in pathFiles):
    pathFiles.remove('fallos.csv')

# este será el dataframe que contendrá la información
bigFile = pandas.DataFrame()
    
# Procesa cada archivo y lo anexa a un solo DataFrame
for path in pathFiles:
    print(path)
    bigFile = bigFile.append(ReadFile(path), ignore_index = True)

# eliminar filas duplicadas
bigFile.drop_duplicates(inplace = True)


#Lectura archivo
data=pd.read_csv('consolidado.csv',  # el archivo
                    sep = ';',         # separador de campos
                    thousands = None,  # separador de miles para números
                    decimal = '.',
                    skiprows = 0,
                    header=0)     # separador de los decimales para números

frames = [data, bigFile]

data = pd.concat(frames)

data=data[data['country']=='US']
bigFile=bigFile[bigFile['country']=='US']


data['semanas_campana'] = ((pd.to_datetime(data['deadline']) - pd.to_datetime(data['launched_at']))/ np.timedelta64(1, 'W')).astype(int)

#Tratamiento de variables
df_categoria = pd.get_dummies(data['category'])
df_country = pd.get_dummies(data['country'])
df_subcategory = pd.get_dummies(data['subcategory'])

frames = [data, df_categoria, df_country, df_subcategory]
data = pd.concat(frames, axis=1)

cols = ['goal', 'semanas_campana', 'art', 'comics', 'crafts',
       'dance', 'design', 'fashion', 'film & video', 'food', 'games',
       'journalism', 'music', 'photography', 'publishing', 'technology',
       'theater', 
        '3d printing', 'academic', 'accessories', 'action', 'animals', 'animation', 'anthologies', 
'apparel', 'apps', 'architecture', 'art books', 'audio', 'bacon', 'blues', 'calendars', 'camera equipment', 'candles', 
'ceramics', 'children\'s books', 'childrenswear', 'chiptune', 'civic design', 'classical music', 'comedy', 
'comic books', 'community gardens', 'conceptual art', 'cookbooks', 'country & folk', 'couture', 'crochet', 
'digital art', 'diy', 'diy electronics', 'documentary', 'drama', 'drinks', 'electronic music', 'embroidery', 
'events', 'experimental', 'fabrication tools', 'faith', 'family', 'fantasy', 'farmer\'s markets', 'farms', 
'festivals', 'fiction', 'fine art', 'flight', 'food trucks', 'footwear', 'gadgets', 'gaming hardware', 'glass', 
'graphic design', 'graphic novels', 'hardware', 'hip-hop', 'horror', 'illustration', 'immersive', 'indie rock', 
'installations', 'interactive design', 'jazz', 'jewelry', 'kids', 'knitting', 'latin', 'letterpress', 
'literary journals', 'literary spaces', 'live games', 'makerspaces', 'metal', 'mixed media', 'mobile games', 
'movie theaters', 'music videos', 'musical', 'narrative film', 'nature', 'nonfiction', 'painting', 'people', 
'performance art', 'performances', 'periodicals', 'pet fashion', 'photo', 'photobooks', 'places', 'playing cards', 
'plays', 'poetry', 'pop', 'pottery', 'print', 'printing', 'product design', 'public art', 'punk', 'puzzles', 
'quilts', 'r&b', 'radio & podcasts', 'ready-to-wear', 'residencies', 'restaurants', 'robots', 'rock', 'romance', 
'science fiction', 'sculpture', 'shorts', 'small batch', 'software', 'sound', 'space exploration', 'spaces', 
'stationery', 'tabletop games', 'taxidermy', 'television', 'textiles', 'thrillers', 'translations', 'typography', 
'vegan', 'video', 'video art', 'video games', 'wearables', 'weaving', 'web', 'webcomics', 'webseries', 'woodworking', 
'workshops', 'world music', 'young adult', 'zines', 'state'
]

data=data[cols]

data.columns=['goal', 'semanas_campana', 'art', 'comics', 'crafts',
       'dance', 'design', 'fashion', 'film \& video', 'food', 'games',
       'journalism', 'music', 'photography', 'publishing', 'technology',
       'theater',
        '3d printing', 'academic', 'accessories', 'action', 'animals', 'animation', 'anthologies', 
'apparel', 'apps', 'architecture', 'art books', 'audio', 'bacon', 'blues', 'calendars', 'camera equipment', 'candles', 
'ceramics', 'children\'s books', 'childrenswear', 'chiptune', 'civic design', 'classical music', 'comedy', 
'comic books', 'community gardens', 'conceptual art', 'cookbooks', 'country \& folk', 'couture', 'crochet', 
'digital art', 'diy', 'diy electronics', 'documentary', 'drama', 'drinks', 'electronic music', 'embroidery', 
'events', 'experimental', 'fabrication tools', 'faith', 'family', 'fantasy', 'farmer\'s markets', 'farms', 
'festivals', 'fiction', 'fine art', 'flight', 'food trucks', 'footwear', 'gadgets', 'gaming hardware', 'glass', 
'graphic design', 'graphic novels', 'hardware', 'hip-hop', 'horror', 'illustration', 'immersive', 'indie rock', 
'installations', 'interactive design', 'jazz', 'jewelry', 'kids', 'knitting', 'latin', 'letterpress', 
'literary journals', 'literary spaces', 'live games', 'makerspaces', 'metal', 'mixed media', 'mobile games', 
'movie theaters', 'music videos', 'musical', 'narrative film', 'nature', 'nonfiction', 'painting', 'people', 
'performance art', 'performances', 'periodicals', 'pet fashion', 'photo', 'photobooks', 'places', 'playing cards', 
'plays', 'poetry', 'pop', 'pottery', 'print', 'printing', 'product design', 'public art', 'punk', 'puzzles', 
'quilts', 'r\&b', 'radio \& podcasts', 'ready-to-wear', 'residencies', 'restaurants', 'robots', 'rock', 'romance', 
'science fiction', 'sculpture', 'shorts', 'small batch', 'software', 'sound', 'space exploration', 'spaces', 
'stationery', 'tabletop games', 'taxidermy', 'television', 'textiles', 'thrillers', 'translations', 'typography', 
'vegan', 'video', 'video art', 'video games', 'wearables', 'weaving', 'web', 'webcomics', 'webseries', 'woodworking', 
'workshops', 'world music', 'young adult', 'zines', 'state'
]

cols=data.columns

data=data[data['state']=='live']
data=data.loc[data["state"] == "live"]

features = list(cols[:161])
data = data.as_matrix()
data = np.matrix(data)

X_test=data[:,:-1]

bigFile['y_pred']=bag1.predict(X_test)

bigFile["y_pred"]= bigFile['y_pred'].map({1: 'Exitoso', 0: 'Fallido'})


bigFile=bigFile[['name', 'blurb', 'goal', 'pledged', 'deadline', 'launched_at', 'backers_count', 'category', 'y_pred']]

camp_exitosas=bigFile[bigFile['y_pred']=='Exitoso']
camp_fallidas=bigFile[bigFile['y_pred']=='Fallido']
camp_exitosas=camp_exitosas.sort_values(by = ['goal'])
camp_fallidas=camp_fallidas.sort_values(by = ['goal'])

camp_exitosas.to_csv("exitos.xls", sep = ";", na_rep = '', index = False)
camp_fallidas.to_csv("fallos.xls", sep = ";", na_rep = '', index = False)