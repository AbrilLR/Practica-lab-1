# Análisis estadístico de la señal 
## Descipción  
Se realizó un código en python con el fin de analizar una señal de electromiografía obtenida de la fuente Physionet.

### Señal y sus variables estadisticas 

Se extrajo una señal fisiologica en la base de datos physionet con ayuda de la libreria WFDB la cual permite extraer los datos de archivos .dat y .hea, los datos se organizaron en un arreglo. 

```python
import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation
from scipy.stats import gaussian_kde
from scipy.stats import norm

archivo = 'C:/Users/ACER/Desktop/Señales lab 1/Señales lab 1/emg_healthy'
registro = wfdb.rdrecord(archivo)
arreglo = registro.p_signal #arreglo numpy

datos = arreglo[:, 0] #columna 0

```
Una vez los datos se encuentran organizados, se pueden calcular los estadisticos descriptivos de la señal. Los estadisticos permiten describir y analizar el comportamiento de la señal.
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
