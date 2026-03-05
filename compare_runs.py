import pandas as pd
import json

# Données RUN 1
data_r1 = {
    'Run': ['RUN 1'] * 22,
    'Epoch': list(range(1, 23)),
    'Train Loss': [0.2115, 0.0901, 0.0833, 0.0157, 0.0082, 0.0023, 0.0012, 0.0006, 0.0005, 0.0004, 0.0004, 0.0003, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
    'Train Acc': [0.9233, 0.9728, 0.9745, 0.9978, 0.9995, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    'Val Loss': [0.1586, 0.1018, 0.1308, 0.1439, 0.1439, 0.1088, 0.0924, 0.1573, 0.1630, 0.1331, 0.2116, 0.2393, 0.1647, 0.2016, 0.2464, 0.2749, 0.1924, 0.3091, 0.4096, 0.3072, 0.3373, 0.3729],
    'Val Acc': [0.9528, 0.9780, 0.9608, 0.9615, 0.9718, 0.9765, 0.9790, 0.9655, 0.9618, 0.9700, 0.9498, 0.9405, 0.9625, 0.9498, 0.9458, 0.9370, 0.9543, 0.9298, 0.9248, 0.9365, 0.9285, 0.9203],
}

# Données RUN 2
data_r2 = {
    'Run': ['RUN 2'] * 23,
    'Epoch': list(range(1, 24)),
    'Train Loss': [0.2153, 0.0844, 0.0625, 0.0156, 0.0061, 0.0033, 0.0028, 0.0009, 0.0007, 0.0005, 0.0004, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
    'Train Acc': [0.9200, 0.9725, 0.9823, 0.9975, 0.9993, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    'Val Loss': [0.1298, 0.1492, 0.1105, 0.1118, 0.1212, 0.1220, 0.1122, 0.1064, 0.1338, 0.1725, 0.1461, 0.1237, 0.1582, 0.1399, 0.2330, 0.1560, 0.1683, 0.1910, 0.1861, 0.3516, 0.3020, 0.2539, 0.2278],
    'Val Acc': [0.9675, 0.9548, 0.9753, 0.9788, 0.9723, 0.9743, 0.9773, 0.9815, 0.9765, 0.9685, 0.9718, 0.9718, 0.9695, 0.9703, 0.9530, 0.9725, 0.9650, 0.9560, 0.9633, 0.9323, 0.9420, 0.9530, 0.9573],
}

df_r1 = pd.DataFrame(data_r1)
df_r2 = pd.DataFrame(data_r2)

# Combiner
df_combined = pd.concat([df_r1, df_r2], ignore_index=True)

print("\n" + "="*100)
print("📊 TABLEAU COMPLET: RUN 1 vs RUN 2")
print("="*100 + "\n")

print(df_combined.to_string(index=False))

# Sauvegarder
df_combined.to_csv('run1_vs_run2_comparison.csv', index=False)
print("\n✅ Tableau sauvegardé: run1_vs_run2_comparison.csv")

# Résumé
print("\n" + "="*100)
print("📈 RÉSUMÉ COMPARATIF")
print("="*100 + "\n")

summary = f"""
RUN 1 (2026-03-03):
  • Best Val Loss: 0.0924 (Epoch 7)
  • Best Val Acc: 97.90%
  • Test Accuracy: 63.95%
  • Test Loss: 1.3293
  • Early Stop: Epoch 22

RUN 2 (2026-03-04):
  • Best Val Loss: 0.1064 (Epoch 8)
  • Best Val Acc: 98.15%
  • Test Accuracy: 57.80% ❌ PIRE!
  • Test Loss: 1.8566 ❌ PIRE!
  • Early Stop: Epoch 23

DIFFÉRENCES:
  • Val Loss: RUN 1 meilleur (-0.014)
  • Val Acc: RUN 2 meilleur (+0.25%)
  • Test Acc: RUN 1 meilleur (+6.15%) ← ÉNORME!
  • Test Loss: RUN 1 meilleur (-0.53)

CONCLUSION:
  ❌ RUN 2 TEST PIRE de 6.15%!
  ✅ PREUVE: Randomness affect résultats ÉNORMÉMENT
  ✅ SOLUTION: Ajouter SEED pour reproductibilité
"""

print(summary)