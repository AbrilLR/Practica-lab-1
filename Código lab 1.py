import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation, norm

archivo = 'C:\\Users\\sebas\\OneDrive\\Escritorio\\Nueva carpeta\\Señales lab 1\\emg_healthy'

registro = wfdb.rdrecord(archivo)
arreglo = registro.p_signal #arreglo numpy

datos = arreglo[:, 0] #columna 0

frecuencia = registro.fs
print("Frecuencia de muestreo:",frecuencia, "Hz")
periodo = 1 / frecuencia
print("Tiempo entre muestras:",periodo, "s")

num_muestras = registro.sig_len

print(f"El número de muestras en el registro es: {num_muestras}")

valor_min = np.min(datos)
valor_max = np.max(datos)

print(f"Valor mínimo: {valor_min}")
print(f"Valor máximo: {valor_max}")

#Gráfica
tiempo = np.arange(0, len(datos))*periodo
plt.figure(figsize=(8, 6))
plt.plot(tiempo, datos)
plt.title("Señal EMG")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje(mV)")
plt.grid(True)
plt.show()

#Media con función 1
media = np.mean(datos)
print("Media1:",media)

#Media con función 2
suma2 = np.sum(datos)
m=len(datos)
media2=suma2/m
print("Media2:",media2)

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


# desviación con función
desv1 = np.std(datos)

# desviación manual:
sumatoria = sum((x - media3) ** 2 for x in datos)#
desv2 = (sumatoria / len(datos)) ** 0.5  

print("Desviación1:",desv1)
print("Desviación2:",desv2)

#Desviación con código
suma=0
i=0
while i < m:
    resta = datos[i] - media3
    cuadrado = resta ** 2
    suma += cuadrado
    i += 1
varianza = suma/(m)
desv3 = varianza ** 0.5
print("Desviación 3:",desv3)

# CV con función
cv1 = variation(datos) * 100
# CV manual
cv2 = (desv2 / media3) * 100  

print("Coeficiente de variación1:",cv1,"%")
print("Coeficiente de variación2:",cv2,"%") 

#Histograma con código
intervalos = np.zeros(17, dtype=int)
i = 0
while i < m:
    dato = datos[i]
    if -0.515 <= dato < -0.4192:
        intervalos[0] += 1
    elif -0.4192 <= dato < -0.3234:
        intervalos[1] += 1
    elif -0.3234 <= dato < -0.2276:
        intervalos[2] += 1
    elif -0.2276 <= dato < -0.1318:
        intervalos[3] += 1
    elif -0.1318 <= dato < -0.0360:
        intervalos[4] += 1
    elif -0.0360 <= dato < 0.0598:
        intervalos[5] += 1
    elif 0.0598 <= dato < 0.1556:
        intervalos[6] += 1
    elif 0.1556 <= dato < 0.2514:
        intervalos[7] += 1
    elif 0.2514 <= dato < 0.3472:
        intervalos[8] += 1
    elif 0.3472 <= dato < 0.4430:
        intervalos[9] += 1
    elif 0.4430 <= dato < 0.5388:
        intervalos[10] += 1
    elif 0.5388 <= dato < 0.6346:
        intervalos[11] += 1
    elif 0.6346 <= dato < 0.7304:
        intervalos[12] += 1
    elif 0.7304 <= dato < 0.8262:
        intervalos[13] += 1
    elif 0.8262 <= dato < 0.9220:
        intervalos[14] += 1
    elif 0.9220 <= dato < 1.0178:
        intervalos[15] += 1
    elif 1.0178 <= dato <= 1.1133:
        intervalos[16] += 1
    i += 1

for i in range(17):
    print("Intervalo",i+1,":",intervalos[i],"datos")
hist, bins = np.histogram(datos, bins=17, density=True)  # Método con función
prob = hist / sum(hist)  # Método manual 

plt.bar(range(17), intervalos, width=1, edgecolor='black', color='b', alpha=0.6)
lim = [valor_min + i * 0.09578235294117646 for i in range(18)]
plt.xticks(range(17),[f"[{lim[i]:.2f}, {lim[i+1]:.2f})" for i in range(17)], rotation=45)
plt.title("Histograma con código señal EMG")
plt.xlabel("Voltaje (mV)")
plt.ylabel("Frecuencia")
plt.grid()
plt.tight_layout()
plt.show()

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

#Ruido Gaussiano
media_r1 = 0         
cv_r1 = 0.01

ruido_g1 = np.random.normal(media_r1, cv_r1, size=len(datos))
datos2= datos + ruido_g1
plt.plot(tiempo, datos2)
plt.title("Señal EMG con ruido gaussiano debil")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje(mV)")
plt.grid(True)
plt.show()

media_r2 = 0         
cv_r2 = 0.50

ruido_g2 = np.random.normal(media_r2, cv_r2, size=len(datos))
datos3= datos + ruido_g2
plt.plot(tiempo, datos3)
plt.title("Señal EMG con ruido gaussiano fuerte")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje(mV)")
plt.grid(True)
plt.show()

snr1 =(10 * np.log10(np.var(datos) / np.var(ruido_g1)))

snr2 =(10 * np.log10(np.var(datos) / np.var(ruido_g2)))

print("SNR1:",snr1)
print("SNR2:",snr2)

#Ruido impulso
num_imp1 = int(0.001 * len(datos))  #porcentaje de la señal con impulsos
pos= np.random.randint(0, len(datos), num_imp1)
ruido_i1 = np.zeros(len(datos))
ruido_i1[pos] = np.random.uniform(-5, 5, num_imp1)  
datos4 = datos + ruido_i1

plt.figure(figsize=(8, 6))
plt.plot(tiempo, datos4)
plt.title("Señal EMG con Ruido impulso bajo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.grid(True)
plt.show()

num_imp2 = int(0.01 * len(datos))  #porcentaje de la señal con impulsos
pos= np.random.randint(0, len(datos), num_imp2)
ruido_i2 = np.zeros(len(datos))
ruido_i2[pos] = np.random.uniform(-5, 5, num_imp2)  
datos5 = datos + ruido_i2

plt.figure(figsize=(8, 6))
plt.plot(tiempo, datos5)
plt.title("Señal EMG con Ruido impulso alto")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.grid(True)
plt.show()

snr3 = (10 * np.log10(np.var(datos) / np.var(ruido_i1)))
snr4 = (10 * np.log10(np.var(datos) / np.var(ruido_i2)))
print("SNR1:",snr3)
print("SNR2:",snr4)

#Ruido Artefacto

ruido_a1 = 0.1 * np.sin(2 * np.pi * 50 * tiempo)
datos5 = datos + ruido_a1

plt.figure(figsize=(8, 6))
plt.plot(tiempo,datos5)
plt.title("Señal EMG con Ruido por Artefacto (50 Hz) debil")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.grid(True)
plt.show()

ruido_a2 = np.sin(2 * np.pi * 50 * tiempo)
datos6 = datos + ruido_a2

plt.figure(figsize=(8, 6))
plt.plot(tiempo,datos6)
plt.title("Señal EMG con Ruido por Artefacto (50 Hz) fuerte")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.grid(True)
plt.show()

snr5 = (10 * np.log10(np.var(datos) / np.var(ruido_a1)))
snr6 = (10 * np.log10(np.var(datos) / np.var(ruido_a2)))
print("SNR1:",snr5)
print("SNR2:",snr6)

