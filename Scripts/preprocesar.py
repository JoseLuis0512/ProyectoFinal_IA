"""
preprocesar.py — Pipeline de preprocesamiento de audio
Proyecto Final IA — Universidad Rafael Landívar
Panel de Domótica Controlada por Voz

Uso:
    python preprocesar.py

Genera en datos_procesados/:
    X_train.npy, X_val.npy, X_test.npy
    y_train.npy, y_val.npy, y_test.npy
    label_encoder.pkl

Requisitos:
    pip install librosa numpy scikit-learn soundfile
"""

import os
import pickle
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
SAMPLE_RATE   = 16000
DURACION_SEG  = 1.5
N_MFCC        = 40        # coeficientes MFCC (mínimo 13, usamos 40 para mejor precisión)
N_FFT         = 512       # tamaño de ventana FFT
HOP_LENGTH    = 160       # salto entre ventanas (10 ms a 16kHz)
N_MELS        = 64        # número de bandas mel (para Mel-Spectrogram alternativo)

CARPETA_CORPUS    = "corpus"
CARPETA_SALIDA    = "datos_procesados"

COMANDOS = [
    "ENCIENDE",
    "APAGA",
    "ALARMA",
    "PERSIANA",
    "TEMPERATURA",
    "DETENTE",
    "RUIDO_FONDO",
]

# Proporciones del dataset
PROP_VAL  = 0.15
PROP_TEST = 0.15

# ──────────────────────────────────────────────
# FUNCIONES DE PREPROCESAMIENTO
# ──────────────────────────────────────────────

def cargar_audio(ruta):
    """Carga un archivo .wav y lo normaliza a SAMPLE_RATE."""
    audio, sr = librosa.load(ruta, sr=SAMPLE_RATE, mono=True)
    return audio

def normalizar_longitud(audio, longitud_objetivo=None):
    """Ajusta el audio a una longitud fija (padding o recorte)."""
    if longitud_objetivo is None:
        longitud_objetivo = int(SAMPLE_RATE * DURACION_SEG)
    if len(audio) < longitud_objetivo:
        audio = np.pad(audio, (0, longitud_objetivo - len(audio)))
    else:
        audio = audio[:longitud_objetivo]
    return audio

def normalizar_amplitud(audio):
    """Normaliza la amplitud del audio entre -1 y 1."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def voice_activity_detection(audio, umbral_energia=0.01):
    """
    VAD simple por energía de trama.
    Retorna el segmento de audio donde hay actividad vocal.
    Si no detecta voz, retorna el audio original.
    """
    tam_trama = int(SAMPLE_RATE * 0.02)  # tramas de 20ms
    energias = []
    for i in range(0, len(audio) - tam_trama, tam_trama):
        trama = audio[i:i + tam_trama]
        energias.append(np.sum(trama ** 2) / tam_trama)

    energias = np.array(energias)
    umbral = umbral_energia * np.max(energias) if len(energias) > 0 else umbral_energia

    tramas_activas = np.where(energias > umbral)[0]
    if len(tramas_activas) == 0:
        return audio  # sin voz detectada, retorna original

    inicio = tramas_activas[0] * tam_trama
    fin    = min((tramas_activas[-1] + 1) * tam_trama, len(audio))
    return audio[inicio:fin]

def extraer_mfcc(audio):
    """
    Extrae los coeficientes MFCC del audio.
    Retorna un array de forma (N_MFCC, T) promediado en el tiempo → (N_MFCC,)
    Para CNN 2D retorna la matriz completa sin promediar.
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    # Delta y delta-delta (capturan dinámica temporal)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Concatenar: forma final (3 * N_MFCC, T)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features

def promediar_features(features):
    """
    Promedia en el tiempo → vector 1D para CNN 1D / MLP / SVM.
    Retorna array de forma (3 * N_MFCC,)
    """
    return np.mean(features, axis=1)

# ──────────────────────────────────────────────
# DATA AUGMENTATION
# ──────────────────────────────────────────────

def augmentar_time_shift(audio, max_shift_ratio=0.2):
    """Desplaza el audio en el tiempo (±20% de la longitud)."""
    shift = int(np.random.uniform(-max_shift_ratio, max_shift_ratio) * len(audio))
    return np.roll(audio, shift)

def augmentar_pitch(audio, n_steps=None):
    """Cambia el tono ±2 semitonos."""
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)

def augmentar_ruido(audio, nivel=0.005):
    """Inyecta ruido gaussiano."""
    ruido = np.random.normal(0, nivel, len(audio))
    return audio + ruido

def augmentar_time_stretch(audio, rate=None):
    """Estira o comprime el audio en el tiempo."""
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(audio, rate=rate)

def aplicar_augmentation(audio):
    """
    Aplica aleatoriamente una combinación de técnicas de augmentation.
    Retorna el audio aumentado.
    """
    tecnicas = [
        augmentar_time_shift,
        augmentar_pitch,
        augmentar_ruido,
        augmentar_time_stretch,
    ]
    # Aplicar 1 o 2 técnicas al azar
    n = np.random.randint(1, 3)
    seleccionadas = np.random.choice(tecnicas, n, replace=False)
    for tecnica in seleccionadas:
        try:
            audio = tecnica(audio)
        except Exception:
            pass  # si falla una técnica, continuar con las demás
    return audio

# ──────────────────────────────────────────────
# PROCESAMIENTO DEL CORPUS
# ──────────────────────────────────────────────

def procesar_corpus(augmentar=True, factor_aug=2):
    """
    Carga todos los audios del corpus, extrae features y aplica augmentation.

    Args:
        augmentar: si True, genera muestras adicionales por augmentation
        factor_aug: número de copias aumentadas por muestra original

    Returns:
        X (array de features), y (etiquetas como strings)
    """
    X, y = [], []
    longitud_objetivo = int(SAMPLE_RATE * DURACION_SEG)

    for comando in COMANDOS:
        carpeta = os.path.join(CARPETA_CORPUS, comando)
        if not os.path.exists(carpeta):
            print(f"  ⚠  Carpeta no encontrada: {carpeta}")
            continue

        archivos = [f for f in os.listdir(carpeta) if f.endswith(".wav")]
        print(f"  Procesando {comando}: {len(archivos)} muestras")

        for archivo in archivos:
            ruta = os.path.join(carpeta, archivo)
            try:
                # Pipeline de preprocesamiento
                audio = cargar_audio(ruta)
                audio = voice_activity_detection(audio)
                audio = normalizar_longitud(audio, longitud_objetivo)
                audio = normalizar_amplitud(audio)

                # Extraer features de la muestra original
                features = extraer_mfcc(audio)
                X.append(promediar_features(features))
                y.append(comando)

                # Augmentation
                if augmentar:
                    for _ in range(factor_aug):
                        audio_aug = aplicar_augmentation(audio.copy())
                        audio_aug = normalizar_longitud(audio_aug, longitud_objetivo)
                        audio_aug = normalizar_amplitud(audio_aug)
                        features_aug = extraer_mfcc(audio_aug)
                        X.append(promediar_features(features_aug))
                        y.append(comando)

            except Exception as e:
                print(f"    ✗ Error en {archivo}: {e}")

    return np.array(X), np.array(y)

# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    print("\n" + "═" * 50)
    print("  PREPROCESAMIENTO DEL CORPUS")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 50)

    print("\n[1/4] Extrayendo features y aplicando augmentation...")
    X, y = procesar_corpus(augmentar=True, factor_aug=2)
    print(f"\n  Total de muestras (con augmentation): {len(X)}")
    print(f"  Dimensión de cada feature vector: {X.shape[1]}")

    # Distribución por clase
    print("\n  Distribución por clase:")
    for cmd in COMANDOS:
        n = np.sum(y == cmd)
        print(f"    {cmd:<15}: {n}")

    print("\n[2/4] Codificando etiquetas...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  Clases: {list(le.classes_)}")

    print("\n[3/4] Dividiendo en train / val / test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=(PROP_VAL + PROP_TEST), random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n[4/4] Guardando archivos...")
    np.save(os.path.join(CARPETA_SALIDA, "X_train.npy"), X_train)
    np.save(os.path.join(CARPETA_SALIDA, "X_val.npy"),   X_val)
    np.save(os.path.join(CARPETA_SALIDA, "X_test.npy"),  X_test)
    np.save(os.path.join(CARPETA_SALIDA, "y_train.npy"), y_train)
    np.save(os.path.join(CARPETA_SALIDA, "y_val.npy"),   y_val)
    np.save(os.path.join(CARPETA_SALIDA, "y_test.npy"),  y_test)

    with open(os.path.join(CARPETA_SALIDA, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\n  ✅ Archivos guardados en '{CARPETA_SALIDA}/'")
    print("\n  Próximo paso: ejecutar entrenar_cnn.py")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()