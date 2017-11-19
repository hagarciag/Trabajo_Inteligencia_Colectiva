
# coding: utf-8

# In[84]:


from sklearn.datasets import load_iris  
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

data=pd.read_csv('consolidado.csv',  # el archivo
                    sep = ';',         # separador de campos
                    thousands = None,  # separador de miles para números
                    decimal = '.',
                    skiprows = 0,
                    header=0)     # separador de los decimales para números

#cols = list(data.loc[:,'goal':'pledged']) + ['backers_count']  + ['category'] + ['subcategory'] + ['state']
cols = list(data.loc[:,'goal':'goal']) + ['backers_count']  + ['category'] + ['subcategory'] + ['state']

data=data[cols]
#data=data[cols].head(5)

# Se categorizan las variables
le2 = preprocessing.LabelEncoder()
le2.fit(data["subcategory"])
data["subcategory"]=le2.transform(data["subcategory"])

le3 = preprocessing.LabelEncoder()
le3.fit(data["category"])
data["category"]=le3.transform(data["category"])

le4 = preprocessing.LabelEncoder()
le4.fit(data["state"])
data["state"]=le4.transform(data["state"])

              
#data.to_csv("dataset_arbol.csv", sep = ";", na_rep = '', index = False)

features = list(cols[:4])
data = data.as_matrix()
data = np.matrix(data)

X_train, X_test, y_train, y_test = train_test_split(
    data[:,:-1], np.ravel(data[:,4:5]), test_size=0.30, random_state=42)

#Se construye el arbol
clf = tree.DecisionTreeClassifier(max_depth=5,max_leaf_nodes=5)
clf = clf.fit(X_train, y_train)

#Se exporta el arbol al formato Graphviz 
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
    
#Poder de prediccion de la clase
y_pred=clf.predict(X_test)

accuracy = sm.accuracy_score(y_test, y_pred)

print("Precision: %.2f%%" % (accuracy*100.0))
print("MSE:",mean_squared_error(y_test, y_pred))

#Probabilidad de que el registro la clase pertenezca a cada una de las hojas
#clf.predict_proba(X_test)

#export_graphviz soporta otras opciones tal como colorear y la funcion Image permite renderizar la imagen en el libro de Ipython
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=features,  
                     class_names=['Exito', 'Fracaso'],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  

