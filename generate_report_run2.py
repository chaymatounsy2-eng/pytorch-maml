from datetime import datetime

report = f"""
╔═══════════════════════════════════════════��════════════════╗
║                                                            ║
║           RAPPORT 2ÈME ENTRAÎNEMENT (RUN 2)               ║
║              Analyse comparative avec Run 1                ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Modèle: MAML avec Conv4 + GlobalAvgPool (SANS MODIFICATIONS)
Dataset: Thermal (Lettuce + Riz → Olive)

═══════════════════════════════════════════════════════════════

1. CONTEXTE

RUN 1 (2026-03-03):
  • Test Accuracy: 63.95%
  • Test Loss: 1.3293
  • Early Stop: Epoch 22

RUN 2 (2026-03-04):
  • IDENTIQUE au RUN 1 (zéro modifications)
  • Objectif: Tester reproductibilité
  • Découverte: RÉSULTATS DIFFÉRENTS! 🎲

RAISON: Randomness (pas de SEED)
  • Poids initialisés aléatoirement
  • Batches mélangés aléatoirement
  • Tâches MAML tirées aléatoirement

═══════════════════════════════════════════════════════════════

2. RÉSULTATS DU RUN 2

2.1 PHASE 1: APPRENTISSAGE (EPOCHS 1-3)

Epoch 1:
  Train Loss: 0.2153, Train Acc: 92.00%
  Val Loss: 0.1298, Val Acc: 96.75%
  
  COMPARAISON RUN 1:
    Train Loss: +0.0038 (pire)
    Train Acc: -0.33% (pire)
    Val Loss: MEILLEUR (-0.0288)! ✅
    Val Acc: MEILLEUR (+1.47%)! ✅
  
  INTERPRÉTATION:
    • Run 2 a batches différents
    • Epoch 1: Val meilleur grâce chance
    • Train légèrement pire

Epoch 2:
  Train Loss: 0.0844, Train Acc: 97.25%
  Val Loss: 0.1492, Val Acc: 95.48%
  
  SIGNAL: Val Loss AUGMENTE! ⚠️
    Run 1: Epoch 2 meilleur (Val Loss = 0.1018)
    Run 2: Epoch 2 pire (Val Loss = 0.1492)
    
  EXPLICATION: Batches plus difficiles

Epoch 3: ✅ MEILLEUR DU RUN 2
  Train Loss: 0.0625, Train Acc: 98.23%
  Val Loss: 0.1105, Val Acc: 97.53%
  
  Model saved! Best: 0.1105

═══════════════════════════════════════════════════════════════

2.2 PHASE 2: AMÉLIORATION MOMENTANÉE (EPOCHS 4-8)

Epoch 8: ⭐ MEILLEUR VAL LOSS du RUN 2
  Train Loss: 0.0009, Train Acc: 100.00%
  Val Loss: 0.1064, Val Acc: 98.15%
  
  COMPARAISON RUN 1:
    Epoch 7 du Run 1 avait Val Loss = 0.0924
    Epoch 8 du Run 2 a Val Loss = 0.1064
    
  = RUN 2 légèrement PIRE

═══════════════════════════════════════════════════════════════

2.3 PHASE 3: OVERFITTING SÉVÈRE (EPOCHS 9-23)

Train Loss: 0.0001, Train Acc: 100.00% (parfait!)
Val Loss: Varie 0.10 → 0.35 (très bruyant!)
Val Acc: Baisse graduellement (98% → 95%)

Même pattern qu'au RUN 1!
  ✓ Train = 100% (mémorisation)
  ✓ Val = fluctue (noise)
  ✓ Overfitting progressif

═══════════════════════════════════════════════════════════════

3. COMPARAISON DIRECTE: RUN 1 vs RUN 2

┌────────────────────────────────────────────────────────────┐
│                RUN 1          RUN 2         DIFFÉRENCE     │
├────────────────────────────────────────────────────────────┤
│ Epoch 1 Val Loss:  0.1586    0.1298    ✓ RUN2 meilleur    │
│ Best Val Loss:     0.0924    0.1064    ❌ RUN1 meilleur    │
│ Best Val Acc:      97.90%    98.15%    ✓ RUN2 meilleur    │
│ Early Stop Epoch:  22        23        ~ Pareil            │
│ Test Loss:         1.3293    1.8566    ❌ RUN1 meilleur    │
│ Test Accuracy:     63.95%    57.80%    ❌ RUN1 meilleur    │
└────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

4. ANALYSE: POURQUOI RUN 2 TEST PIRE?

HYPOTHÈSE 1: Overfitting différent
  • RUN 2 a overfit DIFFÉREMMENT
  • Patterns spécifiques à ses batches
  • Patterns DIFFÉRENTS de RUN 1
  • Olive patterns ne match pas RUN 2 patterns!

HYPOTHÈSE 2: Modèle adapté différemment
  • Poids initiaux différents
  • Apprentissage différent
  • Généralisation différente
  • RUN 1 plus chanceux!

CONCLUSION:
  ❌ RUN 2 test accuracy PIRE (57.80% vs 63.95%)
  ❌ Le modèle est TRÈS sensible au randomness
  ✅ BESOIN de SEED pour stabilité!

═══════════════════════════════════════════════════════════════

5. MÉTRIQUES CLÉS

Val Loss (Best):
  RUN 1: 0.0924 (Epoch 7)
  RUN 2: 0.1064 (Epoch 8)
  Différence: +0.0140

Val Accuracy (Best):
  RUN 1: 97.90% (Epoch 7)
  RUN 2: 98.15% (Epoch 8)
  Différence: +0.25% (RUN 2 légèrement meilleur)

Test Accuracy:
  RUN 1: 63.95%
  RUN 2: 57.80%
  Différence: -6.15% (RUN 1 BEAUCOUP meilleur!)

═══════════════════════════════════════════════════════════════

6. CONCLUSION

DÉCOUVERTE IMPORTANTE:
  Même code → Résultats différents!
  
RAISON: Pas de SEED (randomness)
  • Initialisation aléatoire
  • Batches mélangés aléatoirement
  • Tâches MAML aléatoires

IMPLICATION:
  ❌ Résultats NON reproductibles
  ❌ Difficile d'évaluer vraie performance
  ❌ Test Acc varie énormément (-6% !)

SOLUTION:
  ✅ Ajouter SEED = 42 partout
  ✅ Tous les runs seront identiques
  ✅ Résultats reproductibles!

═══════════════════════════════════════════════════════════════

Fin du rapport RUN 2.
"""

with open('RAPPORT_RUN2_ANALYSE.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("\n✅ Rapport sauvegardé: RAPPORT_RUN2_ANALYSE.txt")