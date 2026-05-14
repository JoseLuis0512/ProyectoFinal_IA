# Proyecto Final IA — Universidad Rafael Landívar

## Panel de Domótica Controlado por Voz

Sistema de domótica inteligente desarrollado como proyecto final del curso de Inteligencia Artificial.  
El proyecto permite controlar dispositivos mediante comandos de voz utilizando procesamiento de audio y modelos de Machine Learning.

---

# Características

- Reconocimiento de comandos de voz
- Procesamiento y clasificación de audio
- Comunicación con Arduino mediante puerto serial
- Preprocesamiento de muestras de audio
- Control domótico básico

---

# Tecnologías Utilizadas

- Python
- PyTorch
- NumPy
- Librosa
- Scikit-learn
- PySerial
- Matplotlib

---

# Instalación

## 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd ProyectoFinal_IA
```

## 2. Instalar dependencias

Para evitar errores de compatibilidad, instalar las librerías en el siguiente orden:

```bash
pip install numpy

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

---

# Corpus de Audio

Esta sección contiene las herramientas necesarias para la creación y procesamiento del corpus de voz utilizado por el modelo.

---

## Grabar Muestras

Ejecutar el siguiente script dentro de la carpeta `Scripts`:

```bash
python grabar.py
```

Este script permite capturar nuevas muestras de voz para entrenamiento.

---

## Preprocesar Audio

Ejecutar:

```bash
python preprocesar.py
```

Este proceso limpia y transforma las muestras de audio para prepararlas para el entrenamiento del modelo.


# Ejecución

Una vez instaladas las dependencias y procesado el corpus, ejecutar el sistema principal:

```bash
python main.py
```

---

# Autores

Proyecto desarrollado para la Universidad Rafael Landívar.

- Jose Enríquez
- [Agregar integrantes]

---

# Licencia

Proyecto desarrollado únicamente con fines educativos.