from sklearn.datasets import load_iris  
from sklearn import tree  
import pydotplus  
from IPython.display import Image  
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

data=pd.read_csv('consolidado.csv',  # el archivo
                    sep = ';',         # separador de campos
                    thousands = None,  # separador de miles para números
                    decimal = '.',
                    skiprows = 0,
                    header=0)     # separador de los decimales para números

#print(data.head(1))

#print(list(list(data.columns.values)))
#print(data.name)
#print(cols)
#print(data[cols].head(50))
#data=data[cols].head(50000)
#Todo esto sportlight con la final
data.drop('spotlight', axis=1)
print(data.columns)
#cols = list(data.loc[:,'goal':'pledged']) + ['backers_count']  + ['category'] + ['subcategory'] + ['state']
cols = list(data.loc[:,'goal':'goal']) + ['backers_count']  + ['category'] + ['subcategory'] + ['state']

data=data[cols]
#data=data[cols].head(5)
print(data.columns)

#le1 = preprocessing.LabelEncoder()
#le1.fit(data["spotlight"])
#data["spotlight"]=le1.transform(data["spotlight"])

le2 = preprocessing.LabelEncoder()
le2.fit(data["subcategory"])
data["subcategory"]=le2.transform(data["subcategory"])

le3 = preprocessing.LabelEncoder()
le3.fit(data["category"])
data["category"]=le3.transform(data["category"])

le4 = preprocessing.LabelEncoder()
le4.fit(data["state"])
data["state"]=le4.transform(data["state"])

              
data.to_csv("dataset_arbol.csv", sep = ";", na_rep = '', index = False)




#features = list(data.columns[:10])
features = list(cols[:4])
data = data.as_matrix()
data = np.matrix(data)

print("Regresores")
#print(data[:,:-1])
print("Objetivo")
#print(np.ravel(data[:,6:7]))
print("antes de la particion")
X_train, X_test, y_train, y_test = train_test_split(
    data[:,:-1], np.ravel(data[:,4:5]), test_size=0.30, random_state=42)

print("despues de la particion")

#Se construye el arbol
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)

print("despues de fit")

#Se exporta el arbol al formato Graphviz 
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
    
print("despues de export_graphviz")

#Con el comando "dot -Tpdf iris.dot -o iris.pdf" se puede crear un PDF del arbol 
#por consola, a partir del archivo iris.dot creado

#Con pydotplus se puede crear el PDF desde el libro son necesidad de hacerlo por linea de comandos
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = pydotplus.graph_from_dot_data(dot_data) 
#graph.write_pdf("iris.pdf")  

#export_graphviz soporta otras opciones tal como colorear y la funcion Image permite renderizar la imagen en el libro de Ipython
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=features,  
                     class_names=['1', '2'],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  

#Poder de prediccion de la clase
#y_pred=clf.predict(X_test)
#print(y_pred)
#print(y_test)
#Probabilidad de que el registro la clase pertenezca a cada una de las hojas
#clf.predict_proba(X_test)
