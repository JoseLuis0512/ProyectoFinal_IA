"""
inferencia_tiempo_real.py — Pipeline completo de inferencia
Proyecto Final IA — Universidad Rafael Landívar
Panel de Domótica Controlada por Voz

Captura audio continuamente, detecta voz (VAD), clasifica el comando
con la CNN y envía el resultado al Arduino por Serial USB.

Uso:
    python scripts/inferencia_tiempo_real.py

    Argumentos opcionales:
        --puerto COM3          (puerto Serial del Arduino, default: auto-detecta)
        --umbral 0.6           (confianza mínima, default: 0.6)
        --sin-arduino          (modo prueba sin hardware)

Requisitos:
    pip install torch librosa sounddevice numpy pyserial
"""

import os
import sys
import time
import pickle
import argparse
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa
import serial
import serial.tools.list_ports

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
SAMPLE_RATE     = 16000
DURACION_SEG    = 1.5
N_MFCC          = 40
N_FFT           = 512
HOP_LENGTH      = 160
BAUDRATE        = 9600
CONFIANZA_MIN   = 0.6

# VAD — detectar inicio de voz
UMBRAL_ENERGIA  = 0.001       # energía mínima para considerar que hay voz
FRAMES_VAD      = 8          # frames consecutivos con voz para activar grabación

CARPETA_MODELOS = "modelos"
DURACION_LSTM   = 2.0    # segundos — audios más largos van al LSTM


# ──────────────────────────────────────────────
# ARQUITECTURA CNN (idéntica a entrenar_cnn.py)
# ──────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.4):
        super(CNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2), nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2), nn.Dropout(dropout * 0.5),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(256), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────
# ARQUITECTURA LSTM (idéntica a entrenar_lstm.py)
# ──────────────────────────────────────────────
class LSTMClasificador(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMClasificador, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
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
# CARGA DEL MODELO
# ──────────────────────────────────────────────
def cargar_modelo():
    ruta_info  = os.path.join(CARPETA_MODELOS, "cnn_comandos_info.pkl")
    ruta_pesos = os.path.join(CARPETA_MODELOS, "cnn_comandos.pt")

    with open(ruta_info, "rb") as f:
        info = pickle.load(f)

    model = CNN1D(input_size=info["input_size"], num_classes=info["num_classes"])
    model.load_state_dict(torch.load(ruta_pesos, map_location="cpu"))
    model.eval()
    return model, info["clases"]


def cargar_modelo_lstm():
    """Carga el modelo LSTM y su metadata. Retorna None si no existe."""
    ruta_info  = os.path.join(CARPETA_MODELOS, "lstm_secuencial_info.pkl")
    ruta_pesos = os.path.join(CARPETA_MODELOS, "lstm_secuencial.pt")
    if not os.path.exists(ruta_info):
        return None, None
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
    return model, info["clases"]


# ──────────────────────────────────────────────
# DETECCIÓN DE PUERTO ARDUINO
# ──────────────────────────────────────────────
def detectar_puerto_arduino():
    """Intenta detectar automáticamente el puerto del Arduino."""
    puertos = serial.tools.list_ports.comports()
    for p in puertos:
        if "Arduino" in p.description or "CH340" in p.description or \
           "USB Serial" in p.description or "ttyUSB" in p.device:
            return p.device
    # Si no detecta, retorna el primero disponible
    if puertos:
        return puertos[0].device
    return None


def conectar_arduino(puerto, sin_arduino=False):
    """Establece conexión Serial con el Arduino."""
    if sin_arduino:
        print("  Modo sin Arduino activado — solo se imprime en consola")
        return None

    if puerto is None:
        puerto = detectar_puerto_arduino()

    if puerto is None:
        print("  ⚠  No se encontró Arduino. Usa --sin-arduino para modo prueba.")
        sys.exit(1)

    try:
        conexion = serial.Serial(puerto, BAUDRATE, timeout=2)
        time.sleep(3)

        conexion.reset_input_buffer()

        respuesta = conexion.readline().decode(errors="ignore").strip()
        if "ARDUINO_LISTO" in respuesta:
            print(f"  ✅ Arduino conectado en {puerto}")
        else:
            print(f"  ⚠  Arduino en {puerto} (respuesta: '{respuesta}')")
        return conexion
    except serial.SerialException as e:
        print(f"  ✗ Error conectando a {puerto}: {e}")
        print("  Usa --sin-arduino para modo prueba.")
        sys.exit(1)


def enviar_comando(arduino, comando):
    """Envía un comando al Arduino por Serial."""
    if arduino is None:
        print(f"  [SIN ARDUINO] → {comando}")
        return
    try:
        arduino.write((comando + "\n").encode())
        time.sleep(0.05)
        # Leer respuesta del Arduino
        while arduino.in_waiting:
            resp = arduino.readline().decode().strip()
            if resp:
                print(f"  Arduino: {resp}")
    except serial.SerialException as e:
        print(f"  ✗ Error enviando comando: {e}")


# ──────────────────────────────────────────────
# PREPROCESAMIENTO DE AUDIO
# ──────────────────────────────────────────────
def preprocesar_audio(audio):
    """Pipeline CNN — promedia MFCC en el tiempo → vector 1D."""
    longitud = int(SAMPLE_RATE * DURACION_SEG)

    # VAD por energía
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

    if len(audio) < longitud:
        audio = np.pad(audio, (0, longitud - len(audio)))
    else:
        audio = audio[:longitud]

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    mfcc   = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE,
                                   n_mfcc=N_MFCC, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return np.mean(features, axis=1)


def preprocesar_audio_lstm(audio, n_frames=128):
    """Pipeline LSTM — conserva secuencia de frames → matriz 2D."""
    longitud = int(SAMPLE_RATE * DURACION_LSTM)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    if len(audio) < longitud:
        audio = np.pad(audio, (0, longitud - len(audio)))
    else:
        audio = audio[:longitud]

    mfcc   = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE,
                                   n_mfcc=N_MFCC, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T  # (T, features)

    if features.shape[0] < n_frames:
        pad = np.zeros((n_frames - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:n_frames, :]

    return features.astype(np.float32)


def predecir_cnn(model, features, clases):
    """Predicción con CNN — entrada 1D."""
    x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()
    idx = probs.argmax().item()
    return clases[idx], probs[idx].item()


def predecir_lstm(model, features, clases):
    """Predicción con LSTM — entrada 2D (frames, features)."""
    x = torch.FloatTensor(features).unsqueeze(0)  # (1, frames, features)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()
    idx = probs.argmax().item()
    return clases[idx], probs[idx].item()


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL CON VAD CONTINUO
# ──────────────────────────────────────────────

cola_audio = queue.Queue()
grabando   = False

def callback_audio(indata, frames, time_info, status):
    """Callback del stream de audio — se ejecuta en hilo separado."""
    if status:
        pass
    cola_audio.put(indata.copy())


def medir_latencia(t_inicio):
    """Retorna la latencia en ms desde t_inicio."""
    return (time.time() - t_inicio) * 1000


def pipeline_inferencia(model_cnn, clases_cnn, model_lstm, clases_lstm,
                        arduino, umbral_confianza):
    """
    Pipeline continuo con dos modelos:
    - Audio corto (< DURACION_LSTM) → CNN → comando simple
    - Audio largo (>= DURACION_LSTM) → LSTM → comando compuesto
    """
    CHUNK        = int(SAMPLE_RATE * 0.02)   # 20ms por chunk
    DURACION_MAX = DURACION_LSTM + 0.5       # máximo de grabación
    buffer_vad   = []
    buffer_rec   = []
    en_comando   = False
    frames_voz   = 0
    frames_silencio = 0
    SILENCIO_FIN = 15   # frames de silencio para cortar grabación
    ultimo_cmd   = ""
    ultimo_t     = 0

    print("\n  🎤 Escuchando... (habla un comando simple o compuesto)")
    print("  Presiona Ctrl+C para salir.\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=CHUNK,
                        callback=callback_audio):
        while True:
            chunk   = cola_audio.get().flatten()
            energia = np.sum(chunk ** 2) / len(chunk)

            if not en_comando:
                if energia > UMBRAL_ENERGIA:
                    frames_voz += 1
                    buffer_vad.append(chunk)
                    if frames_voz >= FRAMES_VAD:
                        en_comando      = True
                        buffer_rec      = buffer_vad.copy()
                        buffer_vad      = []
                        frames_voz      = 0
                        frames_silencio = 0
                        t_inicio        = time.time()
                        print("  🔴 Voz detectada — capturando...")
                else:
                    frames_voz = 0
                    buffer_vad = buffer_vad[-FRAMES_VAD:]

            else:
                buffer_rec.append(chunk)
                duracion = len(buffer_rec) * CHUNK / SAMPLE_RATE

                # Detectar silencio al final para cortar antes del máximo
                if energia <= UMBRAL_ENERGIA:
                    frames_silencio += 1
                else:
                    frames_silencio = 0

                fin_por_silencio = (frames_silencio >= SILENCIO_FIN and
                                    duracion >= DURACION_SEG)
                fin_por_maximo   = duracion >= DURACION_MAX

                if fin_por_silencio or fin_por_maximo:
                    audio       = np.concatenate(buffer_rec)
                    duracion_real = len(audio) / SAMPLE_RATE

                    # Decidir qué modelo usar según duración
                    if duracion_real >= DURACION_LSTM and model_lstm is not None:
                        modelo_usado = "LSTM"
                        features = preprocesar_audio_lstm(audio)
                        clase, confianza = predecir_lstm(model_lstm,
                                                         features, clases_lstm)
                    else:
                        modelo_usado = "CNN"
                        features = preprocesar_audio(audio)
                        clase, confianza = predecir_cnn(model_cnn,
                                                        features, clases_cnn)

                    latencia_ms = medir_latencia(t_inicio)

                    print(f"\n  ── Resultado [{modelo_usado}] ─────────────────")
                    print(f"  Comando   : {clase}")
                    print(f"  Confianza : {confianza:.1%}")
                    print(f"  Duración  : {duracion_real:.2f}s")
                    print(f"  Latencia  : {latencia_ms:.0f} ms")

                    ahora = time.time()
                    if (confianza >= umbral_confianza and
                            clase != "RUIDO_FONDO" and
                            not (clase == ultimo_cmd and ahora - ultimo_t < 2.0)):
                        print(f"  ✅ Enviando: {clase}")
                        enviar_comando(arduino, clase)
                        ultimo_cmd = clase
                        ultimo_t   = ahora
                    elif clase == "RUIDO_FONDO":
                        print(f"  ○  Ruido de fondo — ignorado")
                    elif confianza < umbral_confianza:
                        print(f"  ⚠  Confianza baja ({confianza:.1%}) — ignorado")
                    else:
                        print(f"  ○  Comando repetido — ignorado")

                    print(f"  ────────────────────────────────────────")
                    print(f"\n  🎤 Escuchando...")

                    buffer_rec      = []
                    en_comando      = False
                    frames_silencio = 0


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia en tiempo real — Domótica por Voz"
    )
    parser.add_argument("--puerto",      default=None,  help="Puerto Serial del Arduino (ej: COM3)")
    parser.add_argument("--umbral",      default=0.6,   type=float, help="Confianza mínima (0.0-1.0)")
    parser.add_argument("--sin-arduino", action="store_true", help="Modo prueba sin hardware")
    args = parser.parse_args()

    print("\n" + "═" * 50)
    print("  INFERENCIA EN TIEMPO REAL")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 50)

    print("\n[1/3] Cargando modelos...")
    model_cnn, clases_cnn = cargar_modelo()
    print(f"  CNN  — Clases: {clases_cnn}")

    model_lstm, clases_lstm = cargar_modelo_lstm()
    if model_lstm:
        print(f"  LSTM — Clases: {clases_lstm}")
    else:
        print(f"  LSTM — No encontrado, usando solo CNN")

    print(f"  Umbral de confianza: {args.umbral:.0%}")
    print(f"  Umbral duración LSTM: >{DURACION_LSTM}s")

    print("\n[2/3] Conectando Arduino...")
    arduino = conectar_arduino(args.puerto, args.sin_arduino)

    print("\n[3/3] Iniciando pipeline de audio...")
    try:
        pipeline_inferencia(model_cnn, clases_cnn, model_lstm, clases_lstm,
                            arduino, args.umbral)
    except KeyboardInterrupt:
        print("\n\n  Pipeline detenido.")
        if arduino:
            enviar_comando(arduino, "DETENTE")
            arduino.close()
        print("  ¡Hasta luego!\n")


if __name__ == "__main__":
    main()