"""
grabar_lstm.py — Grabación de frases compuestas para el modelo LSTM
Proyecto Final IA — Universidad Rafael Landívar
Panel de Domótica Controlada por Voz

Graba frases de 2 palabras completas en una sola toma.
El hablante dice la frase completa naturalmente, sin pausas artificiales.

Uso:
    python scripts/grabar_lstm.py

Guarda en corpus_lstm/<CLASE>/<CLASE>_<hablante>_<numero>.wav

Requisitos:
    pip install sounddevice soundfile numpy
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
SAMPLE_RATE   = 16000
DURACION_SEG  = 2.5     # frases compuestas duran más que palabras aisladas
PAUSA_ENTRE   = 1.0     # pausa entre grabaciones
MUESTRAS_META = 60      # meta por clase por hablante

CARPETA_BASE  = "corpus_lstm"

# Frases a grabar con instrucción de pronunciación
FRASES = {
    "ENCIENDE_ALARMA"   : "Di naturalmente: ENCIENDE ALARMA",
    "APAGA_ALARMA"      : "Di naturalmente: APAGA ALARMA",
    "APAGA_PERSIANA"    : "Di naturalmente: APAGA PERSIANA",
    "APAGA_TEMPERATURA" : "Di naturalmente: APAGA TEMPERATURA",
}


# ──────────────────────────────────────────────
# FUNCIONES
# ──────────────────────────────────────────────

def crear_carpetas():
    for frase in FRASES:
        os.makedirs(os.path.join(CARPETA_BASE, frase), exist_ok=True)


def contar_muestras(clase):
    ruta = os.path.join(CARPETA_BASE, clase)
    if not os.path.exists(ruta):
        return 0
    return len([f for f in os.listdir(ruta) if f.endswith(".wav")])


def grabar_muestra():
    muestras = int(SAMPLE_RATE * DURACION_SEG)
    audio = sd.rec(muestras, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def guardar_wav(audio, clase, hablante):
    num    = contar_muestras(clase) + 1
    nombre = f"{clase}_{hablante}_{num:04d}.wav"
    ruta   = os.path.join(CARPETA_BASE, clase, nombre)
    sf.write(ruta, audio, SAMPLE_RATE)
    return ruta


def mostrar_resumen():
    print("\n" + "═" * 50)
    print(f"  {'FRASE':<25}  {'MUESTRAS':>8}  {'META':>6}")
    print("─" * 50)
    for clase in FRASES:
        n    = contar_muestras(clase)
        ok   = "✓" if n >= MUESTRAS_META else " "
        barra = "█" * min(int(n / MUESTRAS_META * 20), 20)
        print(f"  {clase:<25}  {n:>8}  {MUESTRAS_META:>6}  {ok} {barra}")
    print("═" * 50 + "\n")


def seleccionar_frase():
    print("\n¿Qué frase quieres grabar?")
    clases = list(FRASES.keys())
    for i, clase in enumerate(clases, 1):
        n = contar_muestras(clase)
        print(f"  [{i}] {clase:<25} ({n} muestras)")
    print(f"  [0] Salir")
    while True:
        try:
            op = int(input("\nElige una opción: "))
            if op == 0:
                return None
            if 1 <= op <= len(clases):
                return clases[op - 1]
        except ValueError:
            pass
        print("  Opción inválida.")


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    crear_carpetas()

    print("\n" + "═" * 50)
    print("  GRABADOR DE FRASES COMPUESTAS — LSTM")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 50)
    print("\n  IMPORTANTE: Di la frase completa en una sola")
    print("  toma, de forma natural, como si le hablaras")
    print("  a un asistente de voz real.")
    print(f"\n  Duración por muestra: {DURACION_SEG}s")

    hablante = input("\nIngresa tu nombre o ID (sin espacios): ").strip()
    if not hablante:
        hablante = "hablante"

    while True:
        mostrar_resumen()
        clase = seleccionar_frase()
        if clase is None:
            print("\n¡Sesión finalizada!\n")
            mostrar_resumen()
            break

        instruccion = FRASES[clase]
        print(f"\n── Grabando: {clase} ──")
        print(f"   {instruccion}")
        print(f"   Duración: {DURACION_SEG}s por muestra")

        cuantas = input(f"   ¿Cuántas muestras grabar ahora? (Enter = 10): ").strip()
        cuantas = int(cuantas) if cuantas.isdigit() else 10

        print("\n   Prepárate... empezamos en 3 segundos.")
        time.sleep(3)

        grabadas = 0
        for i in range(cuantas):
            print(f"\n   [{i+1}/{cuantas}] 🔴 GRABANDO — {instruccion}")
            audio = grabar_muestra()

            if np.max(np.abs(audio)) < 0.005:
                print("   ⚠  Muestra muy silenciosa, se descartó.")
                continue

            ruta = guardar_wav(audio, clase, hablante)
            grabadas += 1
            print(f"   ✓  Guardado: {os.path.basename(ruta)}")

            if i < cuantas - 1:
                time.sleep(PAUSA_ENTRE)

        print(f"\n   ✅ {grabadas} muestras grabadas.")
        print(f"   Total de '{clase}': {contar_muestras(clase)} muestras.")


if __name__ == "__main__":
    main()