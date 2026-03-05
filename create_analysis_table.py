import pandas as pd
import numpy as np

# Données
epochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

train_loss = np.array([0.2115, 0.0901, 0.0833, 0.0157, 0.0082, 0.0023, 0.0012, 0.0006, 0.0005, 0.0004, 0.0004, 0.0003, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

train_acc = np.array([0.9233, 0.9728, 0.9745, 0.9978, 0.9995, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

val_loss = np.array([0.1586, 0.1018, 0.1308, 0.1439, 0.1439, 0.1088, 0.0924, 0.1573, 0.1630, 0.1331, 0.2116, 0.2393, 0.1647, 0.2016, 0.2464, 0.2749, 0.1924, 0.3091, 0.4096, 0.3072, 0.3373, 0.3729])

val_acc = np.array([0.9528, 0.9780, 0.9608, 0.9615, 0.9718, 0.9765, 0.9790, 0.9655, 0.9618, 0.9700, 0.9498, 0.9405, 0.9625, 0.9498, 0.9458, 0.9370, 0.9543, 0.9298, 0.9248, 0.9365, 0.9285, 0.9203])

# Calculer métriques
gap = train_loss - val_loss

# Détecter overfitting
overfitting_status = []
for g in gap:
    if g < 0.02:
        overfitting_status.append("BON")
    elif g < 0.05:
        overfitting_status.append("MOYEN")
    elif g < 0.10:
        overfitting_status.append("OVERFITTING")
    else:
        overfitting_status.append("SEVERE")

# Créer DataFrame (SANS EMOJIS)
df = pd.DataFrame({
    'Epoch': epochs,
    'Train Loss': np.round(train_loss, 4),
    'Train Acc (%)': np.round(train_acc * 100, 2),
    'Val Loss': np.round(val_loss, 4),
    'Val Acc (%)': np.round(val_acc * 100, 2),
    'Gap (Train-Val)': np.round(gap, 4),
    'Overfitting': overfitting_status,
})

print("\n" + "="*140)
print("TABLEAU D'ANALYSE COMPLETE DE L'ENTRAINEMENT")
print("="*140 + "\n")

print(df.to_string(index=False))

print("\n" + "="*140)
print("RESUME STATISTIQUE")
print("="*140)

print(f"""
MEILLEUR MODELE:
├─ Epoch: 7
├─ Val Loss: {val_loss[6]:.4f} (Minimum)
├─ Val Acc: {val_acc[6]:.2%}
└─ Train Loss: {train_loss[6]:.4f}

PIRE OVERFITTING:
├─ Epoch: 19
├─ Gap: {gap[18]:.4f}
├─ Train Acc: {train_acc[18]:.2%} (Parfait!)
├─ Val Acc: {val_acc[18]:.2%} (Bas!)
└��� Raison: Train memorise, Val stagne

EVOLUTION:
├─ Epochs 1-7: Apprentissage BON (Loss baisse)
├─ Epochs 8-22: Overfitting croissant (Val Loss augmente)
└─ Early Stop: Epoch 22 (patience=15)

TEST RESULTS:
├─ Test Accuracy: 63.95% (MAUVAIS)
├─ Test Loss: 1.3293
├─ Raison: Modele overfit sur train/val
└─ Ne generalise pas sur olive!

PROBLEMES IDENTIFIES:
1. Train Accuracy = 100% (memorisation)
2. Val Accuracy = 92% (ne generalise pas)
3. Gap croissant (overfitting severe)
4. Test Accuracy = 64% (tres mauvais)

RECOMMANDATIONS:
1. Reduire hidden_size: 64 → 32
2. Ajouter OLIVE au training
3. Data augmentation
4. Reduire num_shots: 5 → 3
""")

# Sauvegarder en CSV
df.to_csv('training_analysis.csv', index=False)
print("\nTableau sauvegarde: training_analysis.csv")

# ✅ CORRIGER: Créer HTML avec encodage UTF-8 (SANS EMOJIS)
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Training Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: center;
            font-weight: bold;
        }}
        td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #e8f5e9;
        }}
        .best {{
            background-color: #c8e6c9;
            font-weight: bold;
        }}
        .worst {{
            background-color: #ffccbc;
            font-weight: bold;
        }}
        .summary {{
            margin-top: 30px;
            padding: 20px;
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
        }}
    </style>
</head>
<body>
    <h1>Training Analysis Report</h1>
    <p><strong>Dataset:</strong> THERMAL (Lettuce + Riz training, Olive testing)</p>
    <p><strong>Best Model:</strong> Epoch 7 (Val Loss: {val_loss[6]:.4f})</p>
    <p><strong>Test Accuracy:</strong> 63.95% (Overfitting detected!)</p>
    
    <h2>Training Metrics by Epoch</h2>
    {df.to_html(index=False, classes='data-table')}
    
    <div class="summary">
        <h2>Analysis Summary</h2>
        <ul>
            <li><strong>Best Model:</strong> Epoch 7 with Val Loss = {val_loss[6]:.4f}</li>
            <li><strong>Training Phase:</strong> Epochs 1-7 show learning (Loss decreasing)</li>
            <li><strong>Overfitting Phase:</strong> Epochs 8-22 show overfitting (Val Loss increasing)</li>
            <li><strong>Major Issue:</strong> Train Accuracy reaches 100% (memorization)</li>
            <li><strong>Test Performance:</strong> 63.95% accuracy (poor generalization)</li>
            <li><strong>Root Cause:</strong> Model too large for small dataset</li>
        </ul>
        
        <h3>Recommendations:</h3>
        <ol>
            <li>Reduce hidden_size from 64 to 32</li>
            <li>Add OLIVE data to training set</li>
            <li>Implement data augmentation (rotation, flip)</li>
            <li>Reduce num_shots from 5 to 3</li>
            <li>Use regularization techniques (dropout, L2)</li>
        </ol>
    </div>
</body>
</html>"""

# ✅ Écrire avec encodage UTF-8
with open('training_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Tableau HTML sauvegarde: training_analysis.html")
print("\nDone! Fichiers crees:")
print("  - training_analysis.csv (Excel)")
print("  - training_analysis.html (Navigateur)")