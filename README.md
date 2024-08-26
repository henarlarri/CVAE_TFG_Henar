PROYECTO DE ANÁLISIS DE SINTOMATOLOGÍA DEL ALZHEIMER UTILIZANDO CVAEs

**adni utils.py**: contiene la función load adni stack utiliazada en la carga de datos del conjunto. La función permite la normalización delos datos y 
la obtención de la etiqueta DX simple, que clasifica cada muestra en una de las cuatro categorías (CTL, MCI-S, MCI-C, y AD). Además, incluye índices 
que permiten seleccionar rangos específicos de datos, lo cual es útil para dividir el conjunto de datos en subconjuntos de entrenamiento y prueba.

**data set.py**: actúa como un intermediario entre los archivos principales y adni utils.py, facilitando la carga de datos. Simplifica la interac-
ción entre los diferentes módulos, asegurando que los datos se procesen y se carguen correctamente antes de ser utilizados en el entrenamiento
de los modelos.

**modelos.py**: tiene la implementación de los dos modelos de CVAE utilizados en el proyecto. La función de pérdidas de reconstrucciónes igual para 
ambos modelos, por lo que estos comparten el decodificador definido en la clase de python *Conv3DVAEDecoder* . Sin embargo, difieren en las pérdidas 
de divergencia: el primer modelo utiliza *Conv3DVAEEncoder*, que calcula las pérdidas de divergencia basadas en la Divergencia de Kullback-Leibler (KLD), 
mientras que el segundo modelo implementa *Conv3DVAMDDEncoder*, que computa las pérdidas utilizando la función Discrepancia Media Máxima (MMD).

**mytest.py**: archivo utilizado para el entrenamiento de los modelos. Permite ajustar varios hiperparámetros y seleccionar el decodificadora utilizar, 
lo que facilita la obtención de diferentes resultados al variar las configuraciones del modelo.

**inferencia.py**: archivo utilizado para evaluar los modelos. Incluye el código necesario para cargar el modelo entrenado y ejecutar el bucle de pruebas. 
Además, calcula la precisión del modelo utilizando métodos como la validación cruzada y la matriz de confusión. Por último, este script genera gráficos de 
dispersión para las distintas variables latentes y produce muestras sintéticas al variar los valores de estas variables, lo que permite un análisis más 
profundo del comportamiento del modelo.
