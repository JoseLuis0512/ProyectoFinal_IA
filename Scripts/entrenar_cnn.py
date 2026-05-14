"""
entrenar_cnn.py — Entrenamiento del modelo base CNN
Proyecto Final IA — Universidad Rafael Landívar
Panel de Domótica Controlada por Voz

Uso:
    python scripts/entrenar_cnn.py

Requiere haber ejecutado antes:
    python scripts/preprocesar.py

Genera en modelos/:
    cnn_comandos.pt        ← pesos del mejor modelo
    cnn_comandos_info.pkl  ← metadata (clases, input_size, etc.)

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
CARPETA_DATOS  = "datos_procesados"
CARPETA_MODELOS = "modelos"
CARPETA_METRICAS = "metricas"

# Hiperparámetros
EPOCHS      = 80
BATCH_SIZE  = 32
LR          = 1e-3         # learning rate inicial
DROPOUT     = 0.4          # regularización
PATIENCE    = 15           # early stopping: detener si no mejora en N epochs

# Reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# ARQUITECTURA CNN 1D
# ──────────────────────────────────────────────
class CNN1D(nn.Module):
    """
    Red Neuronal Convolucional 1D para clasificación de comandos de voz.

    Entrada: vector de features MFCC de forma (batch, 1, input_size)
    Salida:  logits de forma (batch, num_classes)

    Arquitectura:
        Conv1d(1, 64, kernel=3)  → ReLU → BatchNorm → MaxPool
        Conv1d(64, 128, kernel=3) → ReLU → BatchNorm → MaxPool
        Conv1d(128, 256, kernel=3) → ReLU → BatchNorm → AdaptiveAvgPool
        Flatten
        Linear(256, 128) → ReLU → Dropout
        Linear(128, num_classes)
    """
    def __init__(self, input_size, num_classes, dropout=0.4):
        super(CNN1D, self).__init__()

        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),

            # Bloque 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),

            # Bloque 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),  # reduce a (batch, 256, 1) sin importar input_size
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────

def cargar_datos():
    """Carga los arrays .npy y el label encoder."""
    X_train = np.load(os.path.join(CARPETA_DATOS, "X_train.npy"))
    X_val   = np.load(os.path.join(CARPETA_DATOS, "X_val.npy"))
    X_test  = np.load(os.path.join(CARPETA_DATOS, "X_test.npy"))
    y_train = np.load(os.path.join(CARPETA_DATOS, "y_train.npy"))
    y_val   = np.load(os.path.join(CARPETA_DATOS, "y_val.npy"))
    y_test  = np.load(os.path.join(CARPETA_DATOS, "y_test.npy"))

    with open(os.path.join(CARPETA_DATOS, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    return X_train, X_val, X_test, y_train, y_val, y_test, le


def preparar_tensores(X_train, X_val, X_test, y_train, y_val, y_test):
    """Convierte arrays numpy a tensores PyTorch y agrega dimensión de canal."""
    # Agregar dimensión de canal: (N, features) → (N, 1, features)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    X_val_t   = torch.FloatTensor(X_val).unsqueeze(1)
    X_test_t  = torch.FloatTensor(X_test).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train)
    y_val_t   = torch.LongTensor(y_val)
    y_test_t  = torch.LongTensor(y_test)

    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t


def entrenar_epoch(model, loader, criterion, optimizer, device):
    """Entrena el modelo por una epoch completa."""
    model.train()
    loss_total, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        loss_total += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += X_batch.size(0)

    return loss_total / total, correct / total


def evaluar(model, loader, criterion, device):
    """Evalúa el modelo sin actualizar pesos."""
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
            total += X_batch.size(0)

    return loss_total / total, correct / total


def graficar_entrenamiento(historia, carpeta):
    """Genera y guarda la gráfica de loss y accuracy por epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(historia["train_loss"]) + 1)

    ax1.plot(epochs, historia["train_loss"], label="Train")
    ax1.plot(epochs, historia["val_loss"],   label="Validación")
    ax1.set_title("Loss por epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, historia["train_acc"], label="Train")
    ax2.plot(epochs, historia["val_acc"],   label="Validación")
    ax2.set_title("Accuracy por epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(carpeta, "training_history_cnn.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  Gráfica guardada: {ruta}")


def graficar_confusion(y_true, y_pred, clases, carpeta):
    """Genera y guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)

    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Matriz de confusión — CNN")
    plt.tight_layout()

    ruta = os.path.join(carpeta, "confusion_matrix_cnn.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  Matriz de confusión guardada: {ruta}")


def guardar_reporte(reporte, carpeta):
    """Guarda el reporte de métricas en un archivo de texto."""
    ruta = os.path.join(carpeta, "reporte_metricas_cnn.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("REPORTE DE MÉTRICAS — MODELO CNN\n")
        f.write("=" * 50 + "\n\n")
        f.write(reporte)
    print(f"  Reporte guardado: {ruta}")


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def main():
    os.makedirs(CARPETA_MODELOS,  exist_ok=True)
    os.makedirs(CARPETA_METRICAS, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "═" * 55)
    print("  ENTRENAMIENTO CNN — DOMÓTICA POR VOZ")
    print("  Universidad Rafael Landívar — IA 2026")
    print("═" * 55)
    print(f"\n  Dispositivo: {device}")

    # ── Cargar datos ──
    print("\n[1/5] Cargando datos...")
    X_train, X_val, X_test, y_train, y_val, y_test, le = cargar_datos()
    clases = list(le.classes_)
    num_classes = len(clases)
    input_size  = X_train.shape[1]

    print(f"  Clases ({num_classes}): {clases}")
    print(f"  Dimensión de entrada: {input_size}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Preparar tensores y DataLoaders ──
    print("\n[2/5] Preparando tensores...")
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = preparar_tensores(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

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

    # ── Construir modelo ──
    print("\n[3/5] Construyendo modelo CNN...")
    model = CNN1D(input_size=input_size, num_classes=num_classes, dropout=DROPOUT)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros entrenables: {total_params:,}")
    print(f"  Arquitectura:\n{model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    # ── Entrenamiento con early stopping ──
    print(f"\n[4/5] Entrenando ({EPOCHS} epochs máx, early stopping={PATIENCE})...")
    print(f"  {'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    print("  " + "─" * 52)

    historia = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_val_loss = float("inf")
    mejor_epoch    = 0
    epochs_sin_mejora = 0
    ruta_mejor_modelo = os.path.join(CARPETA_MODELOS, "cnn_comandos.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = entrenar_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluar(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        historia["train_loss"].append(train_loss)
        historia["train_acc"].append(train_acc)
        historia["val_loss"].append(val_loss)
        historia["val_acc"].append(val_acc)

        # Guardar mejor modelo
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            mejor_epoch    = epoch
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), ruta_mejor_modelo)
            marca = " ✓"
        else:
            epochs_sin_mejora += 1
            marca = ""

        if epoch % 5 == 0 or epoch == 1 or marca:
            print(f"  {epoch:>6} {train_loss:>11.4f} {train_acc:>9.1%} "
                  f"{val_loss:>10.4f} {val_acc:>8.1%}{marca}")

        # Early stopping
        if epochs_sin_mejora >= PATIENCE:
            print(f"\n  Early stopping en epoch {epoch} "
                  f"(mejor epoch: {mejor_epoch})")
            break

    print(f"\n  Mejor modelo: epoch {mejor_epoch}, val_loss={mejor_val_loss:.4f}")

    # ── Evaluación final en test set ──
    print("\n[5/5] Evaluando en conjunto de prueba...")
    model.load_state_dict(torch.load(ruta_mejor_modelo, map_location=device))
    model.eval()

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            y_true_list.extend(y_batch.numpy())
            y_pred_list.extend(predicted.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    acc_test = accuracy_score(y_true, y_pred)
    reporte  = classification_report(y_true, y_pred, target_names=clases)

    print(f"\n  Accuracy en test: {acc_test:.1%}")
    print(f"\n  Reporte por clase:\n{reporte}")

    # ── Guardar gráficas y métricas ──
    graficar_entrenamiento(historia, CARPETA_METRICAS)
    graficar_confusion(y_true, y_pred, clases, CARPETA_METRICAS)
    guardar_reporte(reporte, CARPETA_METRICAS)

    # ── Guardar metadata del modelo ──
    info = {
        "input_size"  : input_size,
        "num_classes" : num_classes,
        "clases"      : clases,
        "label_encoder": le,
        "mejor_epoch" : mejor_epoch,
        "val_loss"    : mejor_val_loss,
        "test_acc"    : acc_test,
    }
    ruta_info = os.path.join(CARPETA_MODELOS, "cnn_comandos_info.pkl")
    with open(ruta_info, "wb") as f:
        pickle.dump(info, f)

    print(f"\n  ✅ Modelo guardado: {ruta_mejor_modelo}")
    print(f"  ✅ Metadata guardada: {ruta_info}")
    print("\n  Próximo paso: ejecutar scripts/entrenar_lstm.py")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()