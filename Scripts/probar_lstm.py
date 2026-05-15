"""
probar_lstm.py — Prueba rápida del modelo LSTM con el micrófono
Proyecto Final IA — Universidad Rafael Landívar

Graba 2.5 segundos de audio, extrae la secuencia MFCC
y predice el comando compuesto.

Uso:
    python scripts/probar_lstm.py

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
# CONFIGURACIÓN (debe coincidir con preprocesar_lstm.py)
# ──────────────────────────────────────────────
SAMPLE_RATE     = 16000
DURACION_SEG    = 2.5
N_MFCC          = 40
N_FFT           = 512
HOP_LENGTH      = 160
N_FRAMES        = 128
CONFIANZA_MIN   = 0.5
CARPETA_MODELOS = "modelos"


# ──────────────────────────────────────────────
# ARQUITECTURA (debe ser idéntica a entrenar_lstm.py)
# ──────────────────────────────────────────────
class LSTMClasificador(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout=0.3):
        super(LSTMClasificador, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


# ──────────────────────────────────────────────
# FUNCIONES
# ──────────────────────────────────────────────

def cargar_modelo():
    """Carga el modelo LSTM y su metadata."""
    ruta_info  = os.path.join(CARPETA_MODELOS, "lstm_secuencial_info.pkl")
    ruta_pesos = os.path.join(CARPETA_MODELOS, "lstm_secuencial.pt")

    if not os.path.exists(ruta_info):
        print("  ✗ No se encontró el modelo LSTM.")
        print("    Ejecuta primero: python scripts/entrenar_lstm.py")
        exit(1)

    with open(ruta_info, "rb") as f:
        info = pickle.load(f)

    model = LSTMClasificador(
        input_size=info["input_size"],
        hidden_size=info["hidden_size"],
        num_layers=info["num_layers"],
        num_classes=info["num_classes"]
    )
    model.load_state_dict(torch.load(ruta_pesos, map_location="cpu"))
    model.eval()

    return model, info["clases"], info.get("n_frames", N_FRAMES)


def grabar_audio():
    """Graba DURACION_SEG segundos desde el micrófono."""
    muestras = int(SAMPLE_RATE * DURACION_SEG)
    audio = sd.rec(muestras, samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def preprocesar_audio(audio, n_frames):
    """Extrae secuencia MFCC — pipeline idéntico a preprocesar_lstm.py."""
    longitud = int(SAMPLE_RATE * DURACION_SEG)

    # Normalizar amplitud
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Normalizar longitud
    if len(audio) < longitud:
        audio = np.pad(audio, (0, longitud - len(audio)))
    else:
        audio = audio[:longitud]

    # Extraer MFCC + delta + delta-delta
    mfcc   = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE,
                                   n_mfcc=N_MFCC, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Transponer: (T, features)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T

    # Padding o recorte a n_frames fijo
    if features.shape[0] < n_frames:
        pad = np.zeros((n_frames - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:n_frames, :]

    return features.astype(np.float32)


def predecir(model, features, clases):
    """Retorna la clase predicha, su probabilidad y todas las probabilidades."""
    x = torch.FloatTensor(features).unsqueeze(0)  # (1, frames, features)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()

    idx       = probs.argmax().item()
    confianza = probs[idx].item()
    return clases[idx], confianza, probs.numpy()


def barra_confianza(prob, ancho=20):
    llenas = int(prob * ancho)
    return "█" * llenas + "░" * (ancho - llenas)


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    print("\n" + "═" * 50)
    print("  PRUEBA DE VOZ — LSTM (comandos compuestos)")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 50)

    print("\nCargando modelo LSTM...")
    model, clases, n_frames = cargar_modelo()

    print(f"\n  Clases reconocidas:")
    for c in clases:
        print(f"    • {c}")
    print(f"\n  Duración de grabación : {DURACION_SEG}s")
    print(f"  Umbral de confianza   : {CONFIANZA_MIN:.0%}")
    print(f"  Frames por secuencia  : {n_frames}")
    print("\n  IMPORTANTE: Di la frase completa en una sola")
    print("  toma. Ejemplo: 'enciende alarma', 'apaga persiana'")
    print("\nPresiona Ctrl+C para salir.\n")

    while True:
        input("  Presiona ENTER para grabar...")
        print(f"  🔴 Grabando {DURACION_SEG}s — di tu frase ahora")

        t_inicio = time.time()
        audio    = grabar_audio()
        features = preprocesar_audio(audio, n_frames)
        clase, confianza, probs = predecir(model, features, clases)
        latencia_ms = (time.time() - t_inicio) * 1000

        print("\n  ── Resultado ───────────────────────────────")
        for i, (c, p) in enumerate(zip(clases, probs)):
            marca = " ◄" if i == probs.argmax() else ""
            print(f"  {c:<25} {barra_confianza(p)} {p:5.1%}{marca}")

        print(f"\n  Latencia total : {latencia_ms:.0f} ms")

        if confianza >= CONFIANZA_MIN:
            print(f"  ✅ Comando compuesto: {clase}  ({confianza:.1%})")
        else:
            print(f"  ⚠  Confianza baja ({confianza:.1%}) — intenta de nuevo")
            print(f"     Habla más claro y cerca del micrófono")

        print("  ────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()