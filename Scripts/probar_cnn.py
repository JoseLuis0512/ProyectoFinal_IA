"""
probar_cnn.py — Prueba rápida del modelo CNN con el micrófono
Proyecto Final IA — Universidad Rafael Landívar

Uso:
    python scripts/probar_cnn.py

Graba 1.5 segundos de audio, extrae features y predice el comando.
Presiona Ctrl+C para salir.
"""

import os
import pickle
import time
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa

# ──────────────────────────────────────────────
# CONFIGURACIÓN (debe coincidir con preprocesar.py)
# ──────────────────────────────────────────────
SAMPLE_RATE    = 16000
DURACION_SEG   = 1.5
N_MFCC         = 40
N_FFT          = 512
HOP_LENGTH     = 160
CARPETA_MODELOS = "modelos"
CONFIANZA_MIN  = 0.6   # umbral mínimo para aceptar una predicción


# ──────────────────────────────────────────────
# ARQUITECTURA (debe ser idéntica a entrenar_cnn.py)
# ──────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.4):
        super(CNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────
# FUNCIONES
# ──────────────────────────────────────────────

def cargar_modelo():
    """Carga el modelo y su metadata."""
    ruta_info  = os.path.join(CARPETA_MODELOS, "cnn_comandos_info.pkl")
    ruta_pesos = os.path.join(CARPETA_MODELOS, "cnn_comandos.pt")

    with open(ruta_info, "rb") as f:
        info = pickle.load(f)

    model = CNN1D(
        input_size=info["input_size"],
        num_classes=info["num_classes"]
    )
    model.load_state_dict(torch.load(ruta_pesos, map_location="cpu"))
    model.eval()

    return model, info["clases"]


def grabar_audio():
    """Graba DURACION_SEG segundos desde el micrófono."""
    muestras = int(SAMPLE_RATE * DURACION_SEG)
    audio = sd.rec(muestras, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def preprocesar_audio(audio):
    """Aplica el mismo pipeline que preprocesar.py."""
    longitud = int(SAMPLE_RATE * DURACION_SEG)

    # VAD simple por energía
    tam_trama = int(SAMPLE_RATE * 0.02)
    energias = [
        np.sum(audio[i:i+tam_trama] ** 2) / tam_trama
        for i in range(0, len(audio) - tam_trama, tam_trama)
    ]
    if energias:
        umbral = 0.01 * max(energias)
        activas = [i for i, e in enumerate(energias) if e > umbral]
        if activas:
            inicio = activas[0] * tam_trama
            fin    = min((activas[-1] + 1) * tam_trama, len(audio))
            audio  = audio[inicio:fin]

    # Normalizar longitud y amplitud
    if len(audio) < longitud:
        audio = np.pad(audio, (0, longitud - len(audio)))
    else:
        audio = audio[:longitud]

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Extraer MFCC + delta + delta-delta
    mfcc   = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE,
                                   n_mfcc=N_MFCC, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return np.mean(features, axis=1)


def predecir(model, features, clases):
    """Retorna la clase predicha y su probabilidad."""
    x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()

    idx        = probs.argmax().item()
    confianza  = probs[idx].item()
    clase      = clases[idx]
    return clase, confianza, probs.numpy()


def barra_confianza(prob, ancho=20):
    """Genera una barra visual de confianza."""
    llenas = int(prob * ancho)
    return "█" * llenas + "░" * (ancho - llenas)


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    print("\n" + "═" * 50)
    print("  PRUEBA DE VOZ — CNN")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 50)

    print("\nCargando modelo...")
    model, clases = cargar_modelo()
    print(f"  Clases: {clases}")
    print(f"  Umbral de confianza: {CONFIANZA_MIN:.0%}")
    print("\nPresiona Ctrl+C para salir.\n")

    while True:
        input("  Presiona ENTER para grabar...")
        print("  🔴 Grabando... habla ahora")

        audio    = grabar_audio()
        features = preprocesar_audio(audio)
        clase, confianza, probs = predecir(model, features, clases)

        print("\n  ── Resultado ──────────────────────────")
        for i, (c, p) in enumerate(zip(clases, probs)):
            marca = " ◄" if i == probs.argmax() else ""
            print(f"  {c:<15} {barra_confianza(p)} {p:5.1%}{marca}")

        print()
        if confianza >= CONFIANZA_MIN:
            print(f"  ✅ Comando: {clase}  ({confianza:.1%} confianza)")
        else:
            print(f"  ⚠  Confianza baja ({confianza:.1%}) — comando no reconocido")
        print("  ───────────────────────────────────────\n")


if __name__ == "__main__":
    main()