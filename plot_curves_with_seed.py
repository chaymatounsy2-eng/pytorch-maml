import matplotlib.pyplot as plt
import numpy as np

# Données AVEC SEED (Epochs 1-17)
epochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

train_loss = np.array([0.2027, 0.0590, 0.0903, 0.0263, 0.0134, 0.0033, 0.0014, 0.0010, 0.0007, 0.0006, 0.0005, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002])

train_acc = np.array([0.9233, 0.9860, 0.9730, 0.9930, 0.9978, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

val_loss = np.array([0.1600, 0.0768, 0.1347, 0.2016, 0.0972, 0.0774, 0.0867, 0.0838, 0.0806, 0.0826, 0.0843, 0.1072, 0.1015, 0.1188, 0.0935, 0.1082, 0.1598])

val_acc = np.array([0.9510, 0.9850, 0.9598, 0.9385, 0.9813, 0.9825, 0.9828, 0.9825, 0.9835, 0.9838, 0.9783, 0.9728, 0.9773, 0.9720, 0.9695, 0.9675, 0.9610])

# Figure 1: Loss and Accuracy (2 subplots)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Subplot 1: Train vs Val Loss
axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2.5, markersize=7)
axes[0].plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2.5, markersize=7)
axes[0].axvline(x=2, color='green', linestyle='--', linewidth=2, label='Best Val Loss (Epoch 2)')
axes[0].axvline(x=17, color='orange', linestyle='--', linewidth=2, label='Early Stop (Epoch 17)')
axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
axes[0].set_title('MAML with SEED=42: Training vs Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')  # Log scale pour voir les petites différences

# Subplot 2: Train vs Val Accuracy
axes[1].plot(epochs, train_acc * 100, 'b-o', label='Train Accuracy', linewidth=2.5, markersize=7)
axes[1].plot(epochs, val_acc * 100, 'r-s', label='Val Accuracy', linewidth=2.5, markersize=7)
axes[1].axvline(x=2, color='green', linestyle='--', linewidth=2, label='Best Val Loss (Epoch 2)')
axes[1].axvline(x=17, color='orange', linestyle='--', linewidth=2, label='Early Stop (Epoch 17)')
axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
axes[1].set_title('MAML with SEED=42: Training vs Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11, loc='lower left')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([85, 102])

plt.tight_layout()
plt.savefig('curves_loss_accuracy_with_seed.png', dpi=300, bbox_inches='tight')
print("✅ Figure sauvegardée: curves_loss_accuracy_with_seed.png")
plt.show()

# Figure 2: Overfitting visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Subplot 1: Gap Train-Val Loss
gap_loss = train_loss - val_loss
colors_loss = ['green' if gap < 0 else 'red' for gap in gap_loss]
axes[0].bar(epochs, gap_loss * 100, color=colors_loss, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Loss Gap (%) [Train - Val]', fontsize=13, fontweight='bold')
axes[0].set_title('Train-Val Loss Gap (Negative = Val better)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Subplot 2: Gap Train-Val Accuracy
gap_acc = (train_acc - val_acc) * 100
colors_acc = ['green' if gap < 1 else 'red' for gap in gap_acc]
axes[1].bar(epochs, gap_acc, color=colors_acc, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Accuracy Gap (%) [Train - Val]', fontsize=13, fontweight='bold')
axes[1].set_title('Train-Val Accuracy Gap (Shows Overfitting)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('overfitting_gap_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Figure sauvegardée: overfitting_gap_visualization.png")
plt.show()

# Figure 3: Detailed Loss Analysis (zoomed in)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: Train Loss (detailed)
axes[0, 0].plot(epochs, train_loss, 'b-o', linewidth=2.5, markersize=7)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Train Loss', fontsize=12)
axes[0, 0].set_title('Train Loss: Rapid Decrease (Overfitting!)', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
for i, (e, l) in enumerate(zip(epochs[::2], train_loss[::2])):
    axes[0, 0].text(e, l, f'{l:.4f}', fontsize=8, ha='center')

# Subplot 2: Val Loss (detailed)
axes[0, 1].plot(epochs, val_loss, 'r-s', linewidth=2.5, markersize=7)
axes[0, 1].axvline(x=2, color='green', linestyle='--', linewidth=2, alpha=0.5)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Val Loss', fontsize=12)
axes[0, 1].set_title('Val Loss: Increases After Epoch 2 (Overfitting!)', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Train Accuracy (detailed)
axes[1, 0].plot(epochs, train_acc * 100, 'b-o', linewidth=2.5, markersize=7)
axes[1, 0].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Perfect (100%)')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Train Accuracy (%)', fontsize=12)
axes[1, 0].set_title('Train Accuracy: Reaches 100% (Memorization!)', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([90, 102])

# Subplot 4: Val Accuracy (detailed)
axes[1, 1].plot(epochs, val_acc * 100, 'r-s', linewidth=2.5, markersize=7)
axes[1, 1].axvline(x=2, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Peak (98.50%)')
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Val Accuracy (%)', fontsize=12)
axes[1, 1].set_title('Val Accuracy: Decreases After Epoch 2 (Overfitting!)', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([93, 100])

plt.tight_layout()
plt.savefig('detailed_loss_accuracy_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Figure sauvegardée: detailed_loss_accuracy_analysis.png")
plt.show()

print("\n" + "="*80)
print("📊 RÉSUMÉ DES COURBES")
print("="*80 + "\n")

summary = f"""
OBSERVATIONS CLÉS:

1. TRAIN LOSS:
   • Epoch 1: 0.2027
   • Epoch 2: 0.0590 (-71%)
   • Epoch 17: 0.0002 (-99.96%)
   → Baisse ÉNORME! Mémorisation!

2. VAL LOSS:
   • Epoch 1: 0.1600
   • Epoch 2: 0.0768 (MEILLEUR!)
   • Epoch 3: 0.1347 (augmente +75%)
   • Epoch 17: 0.1598 (proche du départ)
   → Augmente après epoch 2! Signe overfitting clair!

3. TRAIN ACCURACY:
   • Epoch 1: 92.33%
   • Epoch 6: 100.00% (et reste à 100%!)
   → PARFAIT = MÉMORISATION! ❌

4. VAL ACCURACY:
   • Epoch 1: 95.10%
   • Epoch 2: 98.50% (MEILLEUR!)
   • Epoch 3-17: Baisse graduellement (95.98% → 96.10%)
   → Baisse après peak! Généralisation perdue!

CONCLUSION:
   ✅ Epoch 2 = Point optimal!
   ✅ Après epoch 2 = Overfitting progressif
   ✅ Train 100% mais Val ~96% = Gap énorme!
   ✅ Test 62% = Perte majeure de généralisation
"""

print(summary)