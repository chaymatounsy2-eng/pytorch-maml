import matplotlib.pyplot as plt
import numpy as np

# Données RUN 2
epochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

train_loss_r2 = np.array([0.2153, 0.0844, 0.0625, 0.0156, 0.0061, 0.0033, 0.0028, 0.0009, 0.0007, 0.0005, 0.0004, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

train_acc_r2 = np.array([0.9200, 0.9725, 0.9823, 0.9975, 0.9993, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

val_loss_r2 = np.array([0.1298, 0.1492, 0.1105, 0.1118, 0.1212, 0.1220, 0.1122, 0.1064, 0.1338, 0.1725, 0.1461, 0.1237, 0.1582, 0.1399, 0.2330, 0.1560, 0.1683, 0.1910, 0.1861, 0.3516, 0.3020, 0.2539, 0.2278])

val_acc_r2 = np.array([0.9675, 0.9548, 0.9753, 0.9788, 0.9723, 0.9743, 0.9773, 0.9815, 0.9765, 0.9685, 0.9718, 0.9718, 0.9695, 0.9703, 0.9530, 0.9725, 0.9650, 0.9560, 0.9633, 0.9323, 0.9420, 0.9530, 0.9573])

# RUN 1 (pour comparaison)
train_loss_r1 = np.array([0.2115, 0.0901, 0.0833, 0.0157, 0.0082, 0.0023, 0.0012, 0.0006, 0.0005, 0.0004, 0.0004, 0.0003, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
val_loss_r1 = np.array([0.1586, 0.1018, 0.1308, 0.1439, 0.1439, 0.1088, 0.0924, 0.1573, 0.1630, 0.1331, 0.2116, 0.2393, 0.1647, 0.2016, 0.2464, 0.2749, 0.1924, 0.3091, 0.4096, 0.3072, 0.3373, 0.3729])

# Figure 1: RUN 2 seul
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Loss RUN 2
axes[0].plot(epochs, train_loss_r2, 'b-o', label='Train Loss (RUN 2)', linewidth=2, markersize=6)
axes[0].plot(epochs, val_loss_r2, 'r-s', label='Val Loss (RUN 2)', linewidth=2, markersize=6)
axes[0].axvline(x=8, color='green', linestyle='--', label='Best Val Loss (Epoch 8)', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('RUN 2: Training vs Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Accuracy RUN 2
axes[1].plot(epochs, train_acc_r2 * 100, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
axes[1].plot(epochs, val_acc_r2 * 100, 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
axes[1].axvline(x=8, color='green', linestyle='--', label='Best Val Loss (Epoch 8)', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('RUN 2: Training vs Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([90, 102])

plt.tight_layout()
plt.savefig('run2_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Figure sauvegardée: run2_training_curves.png")
plt.show()

# Figure 2: RUN 1 vs RUN 2 (comparaison)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Val Loss comparison
axes[0].plot(range(1, 23), val_loss_r1, 'b-o', label='RUN 1', linewidth=2, markersize=6)
axes[0].plot(epochs, val_loss_r2, 'r-s', label='RUN 2', linewidth=2, markersize=6)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Val Loss', fontsize=12)
axes[0].set_title('RUN 1 vs RUN 2: Validation Loss Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Test Accuracy comparison
test_data = ['RUN 1', 'RUN 2']
test_acc = [0.6395, 0.5780]
colors = ['green', 'red']

bars = axes[1].bar(test_data, test_acc, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Test Accuracy', fontsize=12)
axes[1].set_title('RUN 1 vs RUN 2: Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].axhline(y=0.50, color='gray', linestyle='--', alpha=0.5, label='50% (random)')
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, test_acc):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('run1_vs_run2_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Figure sauvegardée: run1_vs_run2_comparison.png")
plt.show()