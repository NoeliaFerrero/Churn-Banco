# Churn-Banco
Trabajo Práctico Módulo 2 - Aprendizaje supervisado- El objetivo del proyecto es predecir aquellos clientes con mayor propensión a cancelar la tarjeta de crédito del banco, lo que se conoce como Churn. 
Descripción:

Los datos utilizados corresponden a los clientes de una tarjeta de crédito de un banco. En el mismo podemos identificar las siguientes columnas. 

1.	CLIENTNUM: identificador único del cliente
2.	Attrition_Flag: Indica si el cliente sigue vigente o si ha cancelado la tarjeta 
3.	Customer_Age: edad del cliente
4.	Gender: género del cliente
5.	Dependent_count: cantidad de dependientes que posee el cliente
6.	Education_Level: nivel educativo del cliente
7.	Marital_Status: estado civil del cliente
8.	Income_Category: categorización de ingresos del cliente
9.	Card_Category: tipo de tarjeta que posee el cliente
10.	Months_on_book: cantidad de meses desde que el cliente tiene la tarjeta de crédito
11.	Total_Relationship_Count: Cantidad de productos que tiene el cliente
12.	Months_Inactive_12_mon: Cantidad de meses inactivos en los últimos 12 meses
13.	Contacts_Count_12_mon: Cantidad de veces que se contactó al cliente en los últimos 12 meses
14.	Credit_Limit: límite de crédito de la tarjeta
15.	Total_Revolving_Bal: un saldo renovable es la parte del gasto de la tarjeta de crédito que no se paga al final de un ciclo de facturación.
16.	Avg_Open_To_Buy: Promedio del saldo disponible
17.	Total_Amt_Chng_Q4_Q1: Cambios en los montos de las transacciones 
18.	Total_Trans_Amt: Monto total de transacciones
19.	Total_Trans_Ct: Cantidad total de transacciones
20.	Total_Ct_Chng_Q4_Q1: Cambios en la cantidad de transacciones
21.	Avg_Utilization_Ratio: promedio de utilización de la tarjeta

Consigna:
El objetivo del proyecto es predecir aquellos clientes con mayor propensión a cancelar la tarjeta de crédito del banco, lo que se conoce como Churn. Es decir, predecir el valor de la columna Attrition_Flag
El trabajo debe incluir:
1.	Análisis Exploratorio de Datos

a.	Responder algunas preguntas generales del dominio. Por ejemplo: Cantidad de personas que cancelan la tarjeta según sus ingresos, o dependientes. Edad promedio de los clientes, nivel educativo más frecuente, etc.
b.	Análisis de Valores Nulos. Nota: los nulos se presentan con la palabra “unknown”
c.	Análisis de Outliers
d.	Transformación de variables categóricas. Evaluar cuándo usar one hot encoding o label encoding
e.	Análisis de Distribución y Análisis de correlación de las variables

2.	Dividir los datos en set de entrenamiento y en set de prueba. Trabajar con el set de entrenamiento. Evaluar desbalance de clases, probar hacer un balance un poco más equitativo, probar llevarlo a 50-50. (Oversampling, undersampling o SMOTE)

3.	Entrenar 3 modelos de clasificación. Utilizar técnicas de k-fold y grid search encontrar los mejores hiper parámetros.

4.	Reportar los resultados de cada modelo (Accuracy. Recall, Precision y F-score). Incluir matriz de confusión. Seleccionar el mejor modelo en base al análisis de las métricas.

5.	Este punto es opcional: Incluir algún gráfico de Explainability.
