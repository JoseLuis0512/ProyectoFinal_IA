"""
grabar.py — Script de recolección del corpus de voz
Proyecto Final IA — Universidad Rafael Landívar
Panel de Domótica Controlada por Voz

Uso:
    python grabar.py

Requisitos:
    pip install sounddevice soundfile numpy

El script guarda cada muestra en:
    corpus/<COMANDO>/<COMANDO>_<hablante>_<numero>.wav
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
SAMPLE_RATE    = 16000   # 16 kHz (requerido por el proyecto)
DURACION_SEG   = 1.5     # duración de cada grabación en segundos
MUESTRAS_META  = 50      # muestras por clase por hablante (ajustar según avance)
PAUSA_ENTRE    = 0.8     # segundos de pausa entre grabaciones

COMANDOS = [
    "ENCIENDE",
    "APAGA",
    "ALARMA",
    "PERSIANA",
    "TEMPERATURA",
    "DETENTE",
    "RUIDO_FONDO",
]

INSTRUCCIONES = {
    "ENCIENDE"    : "Di claramente la palabra:  ENCIENDE",
    "APAGA"       : "Di claramente la palabra:  APAGA",
    "ALARMA"      : "Di claramente la palabra:  ALARMA",
    "PERSIANA"    : "Di claramente la palabra:  PERSIANA",
    "TEMPERATURA" : "Di claramente la palabra:  TEMPERATURA",
    "DETENTE"     : "Di claramente la palabra:  DETENTE",
    "RUIDO_FONDO" : "Haz RUIDO, habla de otra cosa, o quédate en silencio",
}

CARPETA_BASE = "corpus"


# ──────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────
def crear_carpetas():
    """Crea la estructura de carpetas del corpus si no existe."""
    for comando in COMANDOS:
        ruta = os.path.join(CARPETA_BASE, comando)
        os.makedirs(ruta, exist_ok=True)


def contar_muestras(comando):
    """Retorna cuántas muestras existen ya para un comando."""
    ruta = os.path.join(CARPETA_BASE, comando)
    if not os.path.exists(ruta):
        return 0
    return len([f for f in os.listdir(ruta) if f.endswith(".wav")])


def siguiente_numero(comando):
    """Retorna el siguiente número de archivo disponible."""
    return contar_muestras(comando) + 1


def grabar_muestra():
    """Graba DURACION_SEG segundos y retorna el array de audio."""
    muestras = int(SAMPLE_RATE * DURACION_SEG)
    audio = sd.rec(muestras, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def guardar_wav(audio, comando, hablante, numero):
    """Guarda el array de audio como archivo .wav."""
    nombre = f"{comando}_{hablante}_{numero:04d}.wav"
    ruta   = os.path.join(CARPETA_BASE, comando, nombre)
    sf.write(ruta, audio, SAMPLE_RATE)
    return ruta


def mostrar_resumen():
    """Muestra el conteo actual de muestras por clase."""
    print("\n" + "═" * 45)
    print(f"  {'COMANDO':<15}  {'MUESTRAS':>8}  {'META':>6}")
    print("─" * 45)
    for cmd in COMANDOS:
        n = contar_muestras(cmd)
        meta = 200 if cmd == "RUIDO_FONDO" else MUESTRAS_META
        barra = "█" * min(int(n / meta * 20), 20)
        estado = "✓" if n >= meta else " "
        print(f"  {cmd:<15}  {n:>8}  {meta:>6}  {estado} {barra}")
    print("═" * 45 + "\n")


def seleccionar_comando():
    """Menú para que el usuario elija qué comando grabar."""
    print("\n¿Qué comando quieres grabar?")
    for i, cmd in enumerate(COMANDOS, 1):
        n = contar_muestras(cmd)
        print(f"  [{i}] {cmd:<15} ({n} muestras grabadas)")
    print(f"  [0] Salir")
    while True:
        try:
            opcion = int(input("\nElige una opción: "))
            if opcion == 0:
                return None
            if 1 <= opcion <= len(COMANDOS):
                return COMANDOS[opcion - 1]
        except ValueError:
            pass
        print("  Opción inválida, intenta de nuevo.")


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────
def main():
    crear_carpetas()

    print("\n" + "═" * 45)
    print("  GRABADOR DE CORPUS — DOMÓTICA POR VOZ")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 45)

    # Identificar al hablante
    hablante = input("\nIngresa tu nombre o ID (sin espacios, ej: juan01): ").strip()
    if not hablante:
        hablante = "hablante"

    while True:
        mostrar_resumen()
        comando = seleccionar_comando()
        if comando is None:
            print("\n¡Sesión finalizada! Resumen guardado.\n")
            mostrar_resumen()
            break

        meta = 200 if comando == "RUIDO_FONDO" else MUESTRAS_META
        print(f"\n── Grabando: {comando} ──")
        print(f"   {INSTRUCCIONES[comando]}")
        print(f"   Duración por muestra: {DURACION_SEG}s  |  Meta: {meta} muestras")

        cuantas = input(f"   ¿Cuántas muestras grabar ahora? (Enter = 10): ").strip()
        cuantas = int(cuantas) if cuantas.isdigit() else 10

        print("\n   Prepárate... empezamos en 3 segundos.")
        time.sleep(3)

        grabadas = 0
        for i in range(cuantas):
            num = siguiente_numero(comando)

            # Señal de inicio
            print(f"\n   [{i+1}/{cuantas}] 🔴 GRABANDO — {INSTRUCCIONES[comando]}")
            audio = grabar_muestra()

            # Verificar que no sea silencio total
            if np.max(np.abs(audio)) < 0.005:
                print("   ⚠  Muestra muy silenciosa, se descartó. Habla más cerca del micrófono.")
                continue

            ruta = guardar_wav(audio, comando, hablante, num)
            grabadas += 1
            print(f"   ✓  Guardado: {os.path.basename(ruta)}")

            if i < cuantas - 1:
                time.sleep(PAUSA_ENTRE)

        print(f"\n   ✅ {grabadas} muestras grabadas para '{comando}'.")
        total = contar_muestras(comando)
        print(f"   Total acumulado de '{comando}': {total} muestras.")


if __name__ == "__main__":
    main()
