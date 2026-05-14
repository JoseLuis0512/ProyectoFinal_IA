"""
generar_corpus_lstm.py — Genera corpus de comandos compuestos
Proyecto Final IA — Universidad Rafael Landívar
Panel de Domótica Controlada por Voz

Concatena pares de audios existentes para crear frases compuestas de 2 palabras.
Ejemplo: ENCIENDE_001.wav + ALARMA_003.wav → "enciende_alarma_001.wav"

Uso:
    python scripts/generar_corpus_lstm.py

Genera en corpus_lstm/:
    ENCIENDE_ALARMA/
    ENCIENDE_PERSIANA/
    ENCIENDE_TEMPERATURA/
    APAGA_ALARMA/
    APAGA_PERSIANA/
    APAGA_TEMPERATURA/
    DETENTE_CERRADURA/

Requisitos:
    pip install numpy soundfile librosa
"""

import os
import random
import numpy as np
import soundfile as sf
import librosa

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
SAMPLE_RATE     = 16000
PAUSA_SEG       = 0.15       # pausa de silencio entre las dos palabras (segundos)
CARPETA_CORPUS  = "corpus"
CARPETA_SALIDA  = "corpus_lstm"
MUESTRAS_META   = 150        # combinaciones a generar por clase
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)

# Definición de comandos compuestos: (palabra1, palabra2) → nombre_clase
COMANDOS_COMPUESTOS = {
    "ENCIENDE_ALARMA"      : ("ENCIENDE",     "ALARMA"),
    "ENCIENDE_PERSIANA"    : ("ENCIENDE",     "PERSIANA"),
    "ENCIENDE_TEMPERATURA" : ("ENCIENDE",     "TEMPERATURA"),
    "APAGA_ALARMA"         : ("APAGA",        "ALARMA"),
    "APAGA_PERSIANA"       : ("APAGA",        "PERSIANA"),
    "APAGA_TEMPERATURA"    : ("APAGA",        "TEMPERATURA"),
}


# ──────────────────────────────────────────────
# FUNCIONES
# ──────────────────────────────────────────────

def listar_audios(comando):
    """Lista todos los .wav disponibles para un comando."""
    carpeta = os.path.join(CARPETA_CORPUS, comando)
    if not os.path.exists(carpeta):
        return []
    return [
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.endswith(".wav")
    ]


def cargar_audio(ruta):
    """Carga y normaliza un audio a SAMPLE_RATE."""
    audio, _ = librosa.load(ruta, sr=SAMPLE_RATE, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def concatenar_con_pausa(audio1, audio2, pausa_seg=PAUSA_SEG):
    """
    Concatena dos audios con una pausa de silencio entre ellos.
    La pausa simula el tiempo natural entre palabras al hablar.
    """
    pausa = np.zeros(int(SAMPLE_RATE * pausa_seg), dtype=np.float32)
    return np.concatenate([audio1, pausa, audio2])


def generar_combinaciones(clase, palabra1, palabra2, n_muestras):
    """
    Genera n_muestras combinaciones aleatorias de audios de palabra1 y palabra2.
    Usa producto cartesiano aleatorio para maximizar variabilidad.
    """
    audios_p1 = listar_audios(palabra1)
    audios_p2 = listar_audios(palabra2)

    if not audios_p1:
        print(f"  ⚠  No se encontraron audios para '{palabra1}'")
        return 0
    if not audios_p2:
        print(f"  ⚠  No se encontraron audios para '{palabra2}'")
        return 0

    carpeta_salida = os.path.join(CARPETA_SALIDA, clase)
    os.makedirs(carpeta_salida, exist_ok=True)

    generadas = 0
    intentos  = 0
    max_intentos = n_muestras * 3

    while generadas < n_muestras and intentos < max_intentos:
        intentos += 1

        # Elegir aleatoriamente un audio de cada palabra
        ruta1 = random.choice(audios_p1)
        ruta2 = random.choice(audios_p2)

        # Evitar combinar el mismo archivo consigo mismo si son iguales
        if palabra1 == palabra2 and ruta1 == ruta2 and len(audios_p1) > 1:
            continue

        try:
            audio1 = cargar_audio(ruta1)
            audio2 = cargar_audio(ruta2)
            combinado = concatenar_con_pausa(audio1, audio2)

            nombre = f"{clase}_{generadas+1:04d}.wav"
            ruta_salida = os.path.join(carpeta_salida, nombre)
            sf.write(ruta_salida, combinado, SAMPLE_RATE)
            generadas += 1

        except Exception as e:
            print(f"    ✗ Error combinando {os.path.basename(ruta1)} + "
                  f"{os.path.basename(ruta2)}: {e}")

    return generadas


def mostrar_resumen():
    """Muestra cuántas muestras se generaron por clase."""
    print("\n" + "═" * 50)
    print(f"  {'CLASE':<25}  {'MUESTRAS':>8}")
    print("─" * 50)
    total = 0
    for clase in COMANDOS_COMPUESTOS:
        carpeta = os.path.join(CARPETA_SALIDA, clase)
        n = len([f for f in os.listdir(carpeta) if f.endswith(".wav")]) \
            if os.path.exists(carpeta) else 0
        total += n
        print(f"  {clase:<25}  {n:>8}")
    print("─" * 50)
    print(f"  {'TOTAL':<25}  {total:>8}")
    print("═" * 50)


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    print("\n" + "═" * 50)
    print("  GENERACIÓN CORPUS LSTM — COMANDOS COMPUESTOS")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 50)
    print(f"\n  Corpus fuente : {CARPETA_CORPUS}/")
    print(f"  Salida        : {CARPETA_SALIDA}/")
    print(f"  Meta por clase: {MUESTRAS_META} combinaciones")
    print(f"  Pausa entre palabras: {PAUSA_SEG*1000:.0f} ms\n")

    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    for clase, (palabra1, palabra2) in COMANDOS_COMPUESTOS.items():
        print(f"  Generando: {clase}  ({palabra1} + {palabra2})")
        n = generar_combinaciones(clase, palabra1, palabra2, MUESTRAS_META)
        print(f"    → {n} combinaciones generadas")

    mostrar_resumen()

    print(f"\n  ✅ Corpus LSTM generado en '{CARPETA_SALIDA}/'")
    print("  Próximo paso: ejecutar scripts/preprocesar_lstm.py")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()