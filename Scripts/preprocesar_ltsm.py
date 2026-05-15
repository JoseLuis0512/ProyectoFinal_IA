"""
preprocesar_lstm.py — Preprocesamiento del corpus para LSTM
Proyecto Final IA — Universidad Rafael Landívar
Panel de Domótica Controlada por Voz

A diferencia del CNN que promedia los MFCC en el tiempo,
el LSTM recibe la secuencia completa de frames MFCC.

Uso:
    python scripts/preprocesar_lstm.py

Genera en datos_procesados/:
    X_seq_train.npy   ← secuencias MFCC de entrenamiento (N, T, features)
    X_seq_val.npy
    X_seq_test.npy
    y_seq_train.npy
    y_seq_val.npy
    y_seq_test.npy
    label_encoder_lstm.pkl

Requisitos:
    pip install librosa numpy scikit-learn soundfile
"""

import os
import pickle
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
SAMPLE_RATE      = 16000
DURACION_SEG     = 2.5       # duración de las grabaciones reales
N_MFCC           = 40
N_FFT            = 512
HOP_LENGTH       = 160
N_FRAMES         = 128       # longitud fija de la secuencia temporal (padding/recorte)

CARPETA_CORPUS   = "corpus_lstm"
CARPETA_SALIDA   = "datos_procesados"

PROP_VAL  = 0.15
PROP_TEST = 0.15

CLASES_LSTM = [
    "ENCIENDE_ALARMA",
    "APAGA_ALARMA",
    "APAGA_PERSIANA",
    "APAGA_TEMPERATURA",
]


# ──────────────────────────────────────────────
# FUNCIONES
# ──────────────────────────────────────────────

def cargar_audio(ruta):
    audio, _ = librosa.load(ruta, sr=SAMPLE_RATE, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def extraer_secuencia_mfcc(audio):
    """
    Extrae la secuencia completa de frames MFCC.
    Retorna matriz de forma (N_FRAMES, 3*N_MFCC) — listo para LSTM.

    A diferencia del CNN que promedia en el tiempo,
    aquí conservamos cada frame para capturar la dinámica temporal.
    """
    mfcc   = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE,
                                   n_mfcc=N_MFCC, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Concatenar: forma (3*N_MFCC, T)
    features = np.concatenate([mfcc, delta, delta2], axis=0)

    # Transponer: (T, 3*N_MFCC) — el LSTM espera (tiempo, features)
    features = features.T

    # Padding o recorte a N_FRAMES fijo
    if features.shape[0] < N_FRAMES:
        pad = np.zeros((N_FRAMES - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:N_FRAMES, :]

    return features.astype(np.float32)


def augmentar_secuencia(audio):
    """Augmentation adaptado para secuencias — solo time shift y ruido."""
    tecnica = np.random.randint(0, 2)
    if tecnica == 0:
        shift = int(np.random.uniform(-0.15, 0.15) * len(audio))
        audio = np.roll(audio, shift)
    else:
        audio = audio + np.random.normal(0, 0.004, len(audio))
    return audio


def procesar_corpus_lstm(augmentar=True, factor_aug=2):
    """Carga todos los audios del corpus LSTM y extrae secuencias MFCC."""
    X, y = [], []
    longitud = int(SAMPLE_RATE * DURACION_SEG)

    for clase in CLASES_LSTM:
        carpeta = os.path.join(CARPETA_CORPUS, clase)
        if not os.path.exists(carpeta):
            print(f"  ⚠  Carpeta no encontrada: {carpeta}")
            continue

        archivos = [f for f in os.listdir(carpeta) if f.endswith(".wav")]
        print(f"  Procesando {clase}: {len(archivos)} muestras")

        for archivo in archivos:
            ruta = os.path.join(carpeta, archivo)
            try:
                audio = cargar_audio(ruta)

                # Normalizar longitud
                if len(audio) < longitud:
                    audio = np.pad(audio, (0, longitud - len(audio)))
                else:
                    audio = audio[:longitud]

                seq = extraer_secuencia_mfcc(audio)
                X.append(seq)
                y.append(clase)

                if augmentar:
                    for _ in range(factor_aug):
                        audio_aug = augmentar_secuencia(audio.copy())
                        seq_aug   = extraer_secuencia_mfcc(audio_aug)
                        X.append(seq_aug)
                        y.append(clase)

            except Exception as e:
                print(f"    ✗ Error en {archivo}: {e}")

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    print("\n" + "═" * 50)
    print("  PREPROCESAMIENTO LSTM")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 50)

    print("\n[1/4] Extrayendo secuencias MFCC...")
    X, y = procesar_corpus_lstm(augmentar=True, factor_aug=2)

    if len(X) == 0:
        print("\n  ✗ No se encontraron datos. Ejecuta primero:")
        print("    python scripts/generar_corpus_lstm.py")
        return

    print(f"\n  Total muestras (con augmentation): {len(X)}")
    print(f"  Forma de cada secuencia: {X.shape[1:]}  (frames × features)")

    print("\n  Distribución por clase:")
    for clase in CLASES_LSTM:
        n = np.sum(y == clase)
        print(f"    {clase:<25}: {n}")

    print("\n[2/4] Codificando etiquetas...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  Clases: {list(le.classes_)}")

    print("\n[3/4] Dividiendo en train / val / test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=(PROP_VAL + PROP_TEST),
        random_state=42,
        stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n[4/4] Guardando archivos...")
    np.save(os.path.join(CARPETA_SALIDA, "X_seq_train.npy"), X_train)
    np.save(os.path.join(CARPETA_SALIDA, "X_seq_val.npy"),   X_val)
    np.save(os.path.join(CARPETA_SALIDA, "X_seq_test.npy"),  X_test)
    np.save(os.path.join(CARPETA_SALIDA, "y_seq_train.npy"), y_train)
    np.save(os.path.join(CARPETA_SALIDA, "y_seq_val.npy"),   y_val)
    np.save(os.path.join(CARPETA_SALIDA, "y_seq_test.npy"),  y_test)

    with open(os.path.join(CARPETA_SALIDA, "label_encoder_lstm.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\n  ✅ Archivos guardados en '{CARPETA_SALIDA}/'")
    print("  Próximo paso: ejecutar scripts/entrenar_lstm.py")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()