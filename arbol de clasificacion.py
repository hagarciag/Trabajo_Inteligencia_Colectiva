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

#Lectura archivo
data=pd.read_csv('consolidado.csv',  # el archivo
                    sep = ';',         # separador de campos
                    thousands = None,  # separador de miles para números
                    decimal = '.',
                    skiprows = 0,
                    header=0)     # separador de los decimales para números

data=data[data['country']=='US']
#Semanas campaña
data['semanas_campana'] = ((pd.to_datetime(data['deadline']) - pd.to_datetime(data['launched_at']))/ np.timedelta64(1, 'W')).astype(int)

#Longitud descripcion
sizes = [len(i) for i in data['blurb'].astype(str)]
#data['longitud_descripcion']=sizes

#Variables dummy de categoria
df_categoria = pd.get_dummies(data['category'])
#data = data.join(df_categoria)

#Variables dummy de country
#data = data.join(df_country)

#Variables dummy de subcategory
df_subcategory = pd.get_dummies(data['subcategory'])
#data = data.join(df_subcategory)

frames = [data, df_categoria, df_subcategory]
data = pd.concat(frames, axis=1)


#Normalizar valor objetivo
#data['goal'] = preprocessing.scale(data['goal'])
#longitud_descripcion
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
# longitud_descripcion
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

data["state"]= data['state'].map({'successful': 1, 'failed': 0})
           
#data.to_csv("dataset_arbol.csv", sep = ";", na_rep = '', index = False)

features = list(cols[:161])
data = data.as_matrix()
data = np.matrix(data)

X_train, X_test, y_train, y_test = train_test_split(
    data[:,:-1], np.ravel(data[:,161:162]), test_size=0.30, random_state=42)

tree1 = DecisionTreeClassifier()

bag = BaggingClassifier(tree1, n_estimators=100, max_samples=0.8, random_state=1)
bag1=bag.fit(X_train, y_train)


y_pred=bag1.predict(X_test)

accuracy = sm.accuracy_score(y_test, y_pred)

print("Precision: %.2f%%" % (accuracy*100.0))
print("MSE:",mean_squared_error(y_test, y_pred))

