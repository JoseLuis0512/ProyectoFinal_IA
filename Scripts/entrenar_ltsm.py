"""
entrenar_lstm.py — Entrenamiento del modelo secuencial LSTM
Proyecto Final IA — Universidad Rafael Landívar
Panel de Domótica Controlada por Voz

Uso:
    python scripts/entrenar_lstm.py

Requiere haber ejecutado antes:
    python scripts/generar_corpus_lstm.py
    python scripts/preprocesar_lstm.py

Genera en modelos/:
    lstm_secuencial.pt
    lstm_secuencial_info.pkl

Requisitos:
    pip install torch numpy scikit-learn matplotlib
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
CARPETA_DATOS    = "datos_procesados"
CARPETA_MODELOS  = "modelos"
CARPETA_METRICAS = "metricas"

EPOCHS      = 80
BATCH_SIZE  = 32
LR          = 1e-3
DROPOUT     = 0.3
PATIENCE    = 15

HIDDEN_SIZE = 128    # neuronas en cada capa LSTM
NUM_LAYERS  = 2      # capas LSTM apiladas

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# ARQUITECTURA LSTM
# ──────────────────────────────────────────────
class LSTMClasificador(nn.Module):
    """
    Red LSTM para clasificación de comandos compuestos de voz.

    Entrada: secuencia de frames MFCC (batch, T, features)
    Salida:  logits (batch, num_classes)

    Arquitectura:
        LSTM(features → hidden_size, num_layers=2, bidireccional)
        Dropout
        Linear(hidden_size*2 → 64)
        ReLU
        Linear(64 → num_classes)

    Bidireccional: procesa la secuencia de izquierda a derecha
    Y también de derecha a izquierda, capturando mejor el contexto
    de cada palabra en la frase.
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout=0.3):
        super(LSTMClasificador, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,        # (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True       # duplica hidden_size en la salida
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),   # *2 por bidireccional
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # output: (batch, seq, hidden*2)
        # hidden: (num_layers*2, batch, hidden)
        output, (hidden, _) = self.lstm(x)

        # Usar el último frame de la secuencia como representación
        ultimo_frame = output[:, -1, :]
        return self.classifier(ultimo_frame)


# ──────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────

def cargar_datos():
    X_train = np.load(os.path.join(CARPETA_DATOS, "X_seq_train.npy"))
    X_val   = np.load(os.path.join(CARPETA_DATOS, "X_seq_val.npy"))
    X_test  = np.load(os.path.join(CARPETA_DATOS, "X_seq_test.npy"))
    y_train = np.load(os.path.join(CARPETA_DATOS, "y_seq_train.npy"))
    y_val   = np.load(os.path.join(CARPETA_DATOS, "y_seq_val.npy"))
    y_test  = np.load(os.path.join(CARPETA_DATOS, "y_seq_test.npy"))

    with open(os.path.join(CARPETA_DATOS, "label_encoder_lstm.pkl"), "rb") as f:
        le = pickle.load(f)

    return X_train, X_val, X_test, y_train, y_val, y_test, le


def entrenar_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_total, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_total += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total   += X_batch.size(0)
    return loss_total / total, correct / total


def evaluar(model, loader, criterion, device):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss_total += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total   += X_batch.size(0)
    return loss_total / total, correct / total


def graficar_entrenamiento(historia, carpeta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(historia["train_loss"]) + 1)
    ax1.plot(epochs, historia["train_loss"], label="Train")
    ax1.plot(epochs, historia["val_loss"],   label="Validación")
    ax1.set_title("Loss por epoch — LSTM")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, historia["train_acc"], label="Train")
    ax2.plot(epochs, historia["val_acc"],   label="Validación")
    ax2.set_title("Accuracy por epoch — LSTM")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta, "training_history_lstm.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  Gráfica guardada: {ruta}")


def graficar_confusion(y_true, y_pred, clases, carpeta):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=True, cmap="Greens")
    ax.set_title("Matriz de confusión — LSTM")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ruta = os.path.join(carpeta, "confusion_matrix_lstm.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  Matriz de confusión guardada: {ruta}")


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    os.makedirs(CARPETA_MODELOS,  exist_ok=True)
    os.makedirs(CARPETA_METRICAS, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "═" * 55)
    print("  ENTRENAMIENTO LSTM — COMANDOS COMPUESTOS")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 55)
    print(f"\n  Dispositivo: {device}")

    print("\n[1/5] Cargando datos...")
    X_train, X_val, X_test, y_train, y_val, y_test, le = cargar_datos()
    clases      = list(le.classes_)
    num_classes = len(clases)
    input_size  = X_train.shape[2]   # features por frame

    print(f"  Clases ({num_classes}): {clases}")
    print(f"  Forma entrada: {X_train.shape[1:]}  (frames × features)")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n[2/5] Preparando tensores...")
    X_train_t = torch.FloatTensor(X_train)
    X_val_t   = torch.FloatTensor(X_val)
    X_test_t  = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t   = torch.LongTensor(y_val)
    y_test_t  = torch.LongTensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE, shuffle=False
    )

    print("\n[3/5] Construyendo modelo LSTM...")
    model = LSTMClasificador(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros entrenables: {total_params:,}")
    print(f"  Arquitectura:\n{model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    print(f"\n[4/5] Entrenando ({EPOCHS} epochs máx, early stopping={PATIENCE})...")
    print(f"  {'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9}")
    print("  " + "─" * 52)

    historia = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_val_loss    = float("inf")
    mejor_epoch       = 0
    epochs_sin_mejora = 0
    ruta_modelo       = os.path.join(CARPETA_MODELOS, "lstm_secuencial.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = entrenar_epoch(model, train_loader,
                                               criterion, optimizer, device)
        val_loss,   val_acc   = evaluar(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        historia["train_loss"].append(train_loss)
        historia["train_acc"].append(train_acc)
        historia["val_loss"].append(val_loss)
        historia["val_acc"].append(val_acc)

        if val_loss < mejor_val_loss:
            mejor_val_loss    = val_loss
            mejor_epoch       = epoch
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), ruta_modelo)
            marca = " ✓"
        else:
            epochs_sin_mejora += 1
            marca = ""

        if epoch % 5 == 0 or epoch == 1 or marca:
            print(f"  {epoch:>6} {train_loss:>11.4f} {train_acc:>9.1%} "
                  f"{val_loss:>10.4f} {val_acc:>8.1%}{marca}")

        if epochs_sin_mejora >= PATIENCE:
            print(f"\n  Early stopping en epoch {epoch} "
                  f"(mejor epoch: {mejor_epoch})")
            break

    print(f"\n  Mejor modelo: epoch {mejor_epoch}, val_loss={mejor_val_loss:.4f}")

    print("\n[5/5] Evaluando en conjunto de prueba...")
    model.load_state_dict(torch.load(ruta_modelo, map_location=device))
    model.eval()

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            y_true_list.extend(y_batch.numpy())
            y_pred_list.extend(predicted.cpu().numpy())

    y_true   = np.array(y_true_list)
    y_pred   = np.array(y_pred_list)
    acc_test = accuracy_score(y_true, y_pred)
    reporte  = classification_report(y_true, y_pred, target_names=clases)

    print(f"\n  Accuracy en test: {acc_test:.1%}")
    print(f"\n  Reporte por clase:\n{reporte}")

    graficar_entrenamiento(historia, CARPETA_METRICAS)
    graficar_confusion(y_true, y_pred, clases, CARPETA_METRICAS)

    ruta_reporte = os.path.join(CARPETA_METRICAS, "reporte_metricas_lstm.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("REPORTE DE MÉTRICAS — MODELO LSTM\n")
        f.write("=" * 50 + "\n\n")
        f.write(reporte)
    print(f"  Reporte guardado: {ruta_reporte}")

    info = {
        "input_size"   : input_size,
        "hidden_size"  : HIDDEN_SIZE,
        "num_layers"   : NUM_LAYERS,
        "num_classes"  : num_classes,
        "clases"       : clases,
        "label_encoder": le,
        "mejor_epoch"  : mejor_epoch,
        "val_loss"     : mejor_val_loss,
        "test_acc"     : acc_test,
        "n_frames"     : X_train.shape[1],
    }
    ruta_info = os.path.join(CARPETA_MODELOS, "lstm_secuencial_info.pkl")
    with open(ruta_info, "wb") as f:
        pickle.dump(info, f)

    print(f"\n  ✅ Modelo guardado: {ruta_modelo}")
    print(f"  ✅ Metadata guardada: {ruta_info}")
    print("\n  Próximo paso: ejecutar scripts/inferencia_tiempo_real.py")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()