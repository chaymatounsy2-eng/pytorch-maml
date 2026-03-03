import matplotlib.pyplot as plt
import json
import numpy as np

# Données extraites de votre training
epochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

train_loss = np.array([0.2115, 0.0901, 0.0833, 0.0157, 0.0082, 0.0023, 0.0012, 0.0006, 0.0005, 0.0004, 0.0004, 0.0003, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

train_acc = np.array([0.9233, 0.9728, 0.9745, 0.9978, 0.9995, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

val_loss = np.array([0.1586, 0.1018, 0.1308, 0.1439, 0.1439, 0.1088, 0.0924, 0.1573, 0.1630, 0.1331, 0.2116, 0.2393, 0.1647, 0.2016, 0.2464, 0.2749, 0.1924, 0.3091, 0.4096, 0.3072, 0.3373, 0.3729])

val_acc = np.array([0.9528, 0.9780, 0.9608, 0.9615, 0.9718, 0.9765, 0.9790, 0.9655, 0.9618, 0.9700, 0.9498, 0.9405, 0.9625, 0.9498, 0.9458, 0.9370, 0.9543, 0.9298, 0.9248, 0.9365, 0.9285, 0.9203])

# ===== FIGURE 1: LOSS =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Loss
axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
axes[0].plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=6)
axes[0].axvline(x=7, color='green', linestyle='--', label='Best Val Loss (Epoch 7)', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Accuracy
axes[1].plot(epochs, train_acc * 100, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
axes[1].plot(epochs, val_acc * 100, 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
axes[1].axvline(x=7, color='green', linestyle='--', label='Best Val Loss (Epoch 7)', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([90, 102])  # Zoom sur la partie intéressante

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Figure sauvegardée: training_curves.png")
plt.show()

# ===== FIGURE 2: OVERFITTING GAP =====
fig, ax = plt.subplots(figsize=(12, 6))

gap = train_loss - val_loss  # Écart train-val

ax.bar(epochs, gap, color=['green' if g < 0.05 else 'orange' if g < 0.10 else 'red' for g in gap], alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Warning (>0.05)')
ax.axhline(y=0.10, color='red', linestyle='--', linewidth=2, label='Overfitting (>0.10)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss Gap (Train - Val)', fontsize=12)
ax.set_title('Overfitting Detection: Train Loss - Val Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('overfitting_gap.png', dpi=300, bbox_inches='tight')
print("✅ Figure sauvegardée: overfitting_gap.png")
plt.show()