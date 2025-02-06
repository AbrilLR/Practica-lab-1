# Análisis estadístico de la señal 
## Descipción  
Se realizó un código en python con el fin de analizar una señal de electromiografía obtenida de la fuente Physionet.

### Señal y sus variables estadisticas 

Se extrajo una señal fisiologica en la base de datos physionet, es necesario contar la libreria WFDB debido a que esta permite extraer los datos de archivos .dat y .hea, los datos se organizaron en un arreglo. 

```python
import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation, norm


archivo = 'C:/Users/ACER/Desktop/Señales lab 1/Señales lab 1/emg_healthy'
registro = wfdb.rdrecord(archivo)
arreglo = registro.p_signal #arreglo numpy

datos = arreglo[:, 0] #columna 0

```
Una vez los datos se encuentran organizados en el arreglo "Datos", se pueden calcular los estadisticos descriptivos de la señal. Los estadisticos permiten describir y analizar el comportamiento de la señal.
Estos valores fueron calculados de dos maneras diferentes, en primer lugar se programó la fómula y posteriormente se utilizaron funciones predefinidas de python.  

``` python 
#Media con código
suma3 = 0
contador = 0
i = 0
while i < m:
    suma3 += datos[i]
    contador += 1
    i += 1
media3=suma3/contador
print("Media3:",media3)
```
``` python

#Desviación estándar con código
suma=0
i=0
while i < m:
    resta = datos[i] - media3
    cuadrado = resta ** 2
    suma += cuadrado
    i += 1
varianza = suma/(m)
desv3 = varianza ** 0.5
print("Desviación estandar 3:",desv3)


```
``` python 
# Coeficiente de variación con código
cv2 = (desv3/media3) * 100  
print("Coeficiente de variación 2:",cv2,"%")

```
## Histogramas 
El histograma permite visualizar la distribucion de la señal. Este gráfico muestra la frecuencia con la que ocurren los valores de la señal dentro de diferentes intervalos.
La función  np.histogram de la libreria NumPy se permite  calcular el histograma de un conjunto de datos. Devuelve la frecuencia de los datos numéricos en intervalos de rangos agrupados.
Además, Genera la función de densidad de probabilidad de una distribución normal para comparar la señal con una distribución ideal.


``` python
plt.figure(figsize=(10, 6))
count, bins, _ = plt.hist(datos, bins=17, alpha=0.6, color='g', edgecolor='black', label="Histograma (Frecuencia)")
x = np.linspace(min(datos), max(datos), 45000)
y = norm.pdf(x, media, desv1)  

scale_factor = max(count) / max(y)  
y_scaled = y * scale_factor

plt.plot(x, y_scaled, color='red', linewidth=2, label="Campana de Gauss ajustada")

plt.title('Histograma de Frecuencia y Ajuste Gaussiano de la Señal EMG')
plt.xlabel('Voltaje (mV)')
plt.ylabel('Frecuencia')  # Se muestra la frecuencia en vez de densidad
plt.grid(True)
plt.legend()
plt.show()

``` 
![HISTOGRAMA](https://github.com/user-attachments/assets/94b42ec3-7e43-462c-bdcd-1df8312586ae)


## Ruido Gaussiano 
ste paso introduce ruido gaussiano en la señal EMG. El ruido gaussiano se utiliza comúnmente para simular interferencias o errores en los datos.


``` python

#Ruido Gaussiano
media_r1 = 0         
cv_r1 = 0.01

ruido_g1 = np.random.normal(media_r1, cv_r1, size=len(datos))

``` 

np.random.normal: Genera ruido con distribución normal. Se simula un ruido débil (con bajo coeficiente de variación) y fuerte (con mayor coeficiente de variación) y se añade a la señal original.



## Ruido Impulso 
En este paso, se introduce ruido impulso en la señal. Este tipo de ruido se presenta como picos o saltos repentinos en los valores de la señal.


``` python
#Ruido impulso
num_imp1 = int(0.001 * len(datos))  #porcentaje de la señal con impulsos
pos= np.random.randint(0, len(datos), num_imp1)
ruido_i1 = np.zeros(len(datos))
ruido_i1[pos] = np.random.uniform(-5, 5, num_imp1)  
datos4 = datos + ruido_i1

``` 
np.random.uniform: Genera valores aleatorios en un intervalo uniforme para simular los picos de impulso.

## Ruido Artefacto 

## Relación señal ruido (SNR)
La relación señal-ruido (SNR) es una medida importante para evaluar la calidad de la señal. SNR es la proporción entre la potencia de la señal útil y la potencia del ruido no deseado.

### Librerias
* wfdb
* numpy
* matplotlib
* scipy

