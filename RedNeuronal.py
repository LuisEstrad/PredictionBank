import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import random


#LEER DATOS LOS DATOS DEL ARCHIVO
datos = pd.read_csv("BaseDatos.csv")
dataframe = pd.DataFrame(datos)

#MOSTRAR LOS DATOS DEL ARCHIVO
#print(datos)

#GUARDAR LOS DATOS DIVIDIENDO ENTRE DATOS PARA PREDECIR Y EL RESULTADO
X = (dataframe[["Cuenta","Se tarda en pagar","Trabajo","Casa Propia","Auto","Deudas","Hijos"]])
y = (dataframe[["Resultado"]])

#ENTRENAMIENTO
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
modelo = LogisticRegression()
modelo.fit(X_train,y_train.values.ravel())

#PREDECIR EL PRESTAMO EN BASE A LAS CARACTERISTICAS
prestamo = pd.DataFrame(X,columns = ['Cuenta','Se tarda en pagar','Trabajo','Casa Propia','Auto','Deudas','Hijos'])
prediccion = modelo.predict(X_test)

#CALCULAR ERROR
mse = mean_squared_error(y_test, prediccion)

#MOSTRAR RESULTADOS
print("==================================================================")
print("REGRESION LOGISTICA: PRIMER CONJUNTO DE DATOS:")
print("Los datos a tomar en cuenta para los prestamos son:")
print(prestamo)
print("  ")
print("La regresion logistica predice un:")
print(prediccion)
print(" ")
print("El resultado verdadero es:")
print(y)
print(" ")
print("El error es:")
print(mse)

#GUARDANDO EL MODELO
filename = 'ModeloFinalizado.sav'
pickle.dump(modelo, open(filename, 'wb'))

#CARGAR DE NUEVO EL MODELO PARA VOLVER A PROBAR CON NUEVOS DATOS
loaded_model1 = pickle.load(open('ModeloFinalizado.sav', 'rb'))

#CREANDO UN NUEVO CONJUNTO DE DATOS, ESTA VEZ DE MANERA ALEATORIA PARA DARLE VARIEDAD
a = random.randint(0, 1)
b = random.randint(0, 1)
c = random.randint(0, 1)
d = random.randint(0, 1)
e = random.randint(0, 1)
f = random.randint(0, 1)
g = random.randint(0, 1)

datanew2 = {'Cuenta' : [a,b,c,d,e,f,g],'Se tarda en pagar' : [g,a,b,c,d,e,f]
           ,'Trabajo':[f,g,a,b,c,d,e], 'Casa Propia':[e,f,g,a,b,c,d]
           ,'Auto':[d,e,f,g,a,b,c], 'Deudas':[c,d,e,f,g,a,b],
           'Hijos':[b,c,d,e,f,g,a]}

resultado2 = [g,f,e,d,c,b,a]

#PREDECIR EL PRESTAMO EN BASE A LAS CARACTERISTICAS
prestamo2 = pd.DataFrame(datanew2,columns = ['Cuenta','Se tarda en pagar','Trabajo','Casa Propia','Auto','Deudas','Hijos'])
predicted = loaded_model1.predict(prestamo2)

#PREDICIENDO EL ERROR
mse2 = mean_squared_error(resultado2, predicted)

#MOSTRAR RESULTADOS
print("==================================================================")
print("REGRESION LOGISTICA: SEGUNDO CONJUNTO DE DATOS:")
print("Los datos a tomar en cuenta para los prestamos son::")
print(prestamo2)
print("  ")
print("La regresion logistica predice un:")	
print(predicted)
print(" ")
print("El resultado verdadero es:")
print(resultado2)
print(" ")
print("El error es:")
print(mse2)
