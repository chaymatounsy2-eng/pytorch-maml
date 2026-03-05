rapport = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     RAPPORT: TEST hidden_size = 32 (RÉSULTATS NÉGATIFS)       ║
║              Analyse détaillée et conclusions                  ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

Date: 2026-03-05
Modification: hidden_size 64 → 32
Test Accuracy: 58.47% (PIRE que 62.02!)
Test Loss: 1.1253

═══════════════════════════════════════════════════════════════════

1. RÉSULTATS CLÉS

BASELINE (hidden=64):
  ✓ Test Accuracy: 62.02%
  ✓ Test Loss: 0.9274
  ✓ Best Val Loss: 0.0768 (Epoch 2)

APRÈS MODIFICATION (hidden=32):
  ✗ Test Accuracy: 58.47% (PIRE!)
  ✗ Test Loss: 1.1253 (PIRE!)
  ✗ Best Val Loss: 0.1111 (Epoch 3, PIRE!)

CHANGEMENT:
  ❌ Test Acc: -3.55% (NÉGATIF!)
  ❌ Test Loss: +0.1979 (PIRE!)
  ❌ Val Loss: +0.0343 (PIRE!)

VERDICT: ❌ MODIFICATION A ÉCHOUÉ!

═══════════════════════════════════════════════════════════════════

2. COMPARAISON DÉTAILLÉE

2.1 TRAINING DYNAMICS

Epoch 1:
  hidden=64: Train Loss=0.2027, Train Acc=92.33%
  hidden=32: Train Loss=0.2516, Train Acc=90.50%
  
  OBSERVATION: hidden=32 start PLUS HAUT (pire!)

Epoch 2:
  hidden=64: Train Loss=0.0590, Train Acc=98.60%
  hidden=32: Train Loss=0.1208, Train Acc=96.38%
  
  OBSERVATION: hidden=32 improvement MOINS rapide!

Epoch 3:
  hidden=64: Train Loss=0.0903, Train Acc=97.30%
  hidden=32: Train Loss=0.0370, Train Acc=99.33%
  
  OBSERVATION: hidden=32 oscille!

Epoch 4+:
  hidden=64: Train Loss → 0.0002, Train Acc → 100%
  hidden=32: Train Loss → 0.0002, Train Acc → 100%
  
  OBSERVATION: Les deux atteignent 100% (mémorisation)

═════════════════════════════════════════════════════════════════

2.2 VALIDATION DYNAMICS

Epoch 1:
  hidden=64: Val Loss=0.1600, Val Acc=95.10%
  hidden=32: Val Loss=0.2011, Val Acc=92.85%
  
  OBSERVATION: hidden=32 PLUS HAUT dès le départ!

Epoch 2:
  hidden=64: Val Loss=0.0768, Val Acc=98.50% ⭐
  hidden=32: Val Loss=0.1349, Val Acc=96.63%
  
  OBSERVATION: hidden=64 meilleur de 0.0581!

Epoch 3:
  hidden=64: Val Loss=0.1347, Val Acc=95.98%
  hidden=32: Val Loss=0.1111, Val Acc=97.35% ⭐
  
  OBSERVATION: hidden=32 meilleur UNIQUEMENT ici!

Epoch 4-18:
  hidden=64: Val Loss varie 0.08-0.16
  hidden=32: Val Loss monte à 0.25-0.50 (ÉNORME!)
  
  OBSERVATION: hidden=32 VBL LOSS AUGMENTE BEAUCOUP!

═════════════════════════════════════════════════════════════════

2.3 TEST PERFORMANCE

Test Accuracy:
  hidden=64: 62.02% ✓ (meilleur)
  hidden=32: 58.47% ❌ (pire)
  Écart: -3.55% (SIGNIFIANT!)

Test Loss:
  hidden=64: 0.9274 ✓ (meilleur)
  hidden=32: 1.1253 ❌ (pire)
  Écart: +0.1979 (SIGNIFIANT!)

INTERPRETATION:
  ❌ hidden=32 TOUT PLUS MAUVAIS!
  ❌ Sur train, val, ET test!
  ❌ Pas amélioré l'overfitting!
  ❌ Même créé underfitting!

═══════════════════════════════════════════════════════════════════

3. ANALYSE: POURQUOI hidden=32 PIRE?

HYPOTHÈSE 1: CAPACITÉ INSUFFISANTE
  • hidden=32: ~100k poids
  • hidden=64: ~400k poids
  • Ratio poids/images:
    - hidden=32: 100k/387 = 258 poids/image
    - hidden=64: 400k/387 = 1034 poids/image
  
  MAIS: Même 258 poids/image devrait suffire!
  
  ANALYSE:
    • Val Loss dès epoch 1 PLUS HAUT
    • Train Loss dès epoch 1 PLUS HAUT
    • → Modèle apprend MOINS BIEN
    • → Underfitting probable!

HYPOTHÈSE 2: OVERFITTING TOUJOURS LÀ
  • hidden=32 AUSSI: Train Acc = 100%
  • Val Loss augmente énormément (0.1111 → 0.5042)
  • Test Acc = 58.47% (terrible)
  
  ANALYSE:
    • Pas résolu overfitting!
    • Combiné avec underfitting!
    • Le pire des deux mondes!

HYPOTHÈSE 3: SEED=42 PAS OPTIMAL POUR hidden=32
  • SEED=42 initialisé pour hidden=64
  • Peut ne pas être optimal pour hidden=32
  • Différent nombre de couches/filtres
  
  MAIS: Reste que résultats PIRES!

═══════════════════════════════════════════════════════════════════

4. CONCLUSION: hidden_size = 32 ÉCHOUE

❌ RÉSULTAT:
   • Test Acc baisse: 62.02% → 58.47%
   • Test Loss augmente: 0.9274 → 1.1253
   • Val Loss augmente: 0.0768 → 0.1111
   • TOUS les metrics PIRES!

❌ SIGNIFICATION:
   • Réduire hidden_size n'a PAS aidé
   • Au lieu d'améliorer, a empiré
   • N'a pas résolu overfitting
   • A créé underfitting AUSSI

⚠️ LEÇON APPRISE:
   • La solution n'est PAS de réduire modèle
   • Problème est OVERFITTING, pas taille
   • Solution doit être ailleurs:
     - Data augmentation
     - Regularization
     - Ajouter données
     - Etc.

═══════════════════════════════════════════════════════════════════

5. RECOMMANDATION: NEXT STEPS

1️⃣ REVENIR À hidden=64 (meilleur jusqu'à présent)
   ✓ 62.02% baseline
   ✓ Mieux que 58.47%!

2️⃣ AJOUTER DATA AUGMENTATION (PRIORITÉ!)
   • Tourner images
   • Flip horizontal
   • Variations légères
   • Devrait aider overfitting
   • Tester avec hidden=64!

3️⃣ AUGMENTER PATIENCE (15 → 25)
   • Laisser plus d'epochs
   • Éviter arrêt prématuré
   • Combiner avec augmentation

4️⃣ TESTER hidden=48 (OPTIONNEL)
   • Compromis entre 32 et 64
   • Si temps/envie
   • Peut être Goldilocks!

═══════════════════════════════════════════════════════════════════

FINAL VERDICT:

❌ hidden_size = 32: MODIFICATION ÉCHOUÉE
   • Pire résultats que baseline
   • N'a pas amélioré overfitting
   • Créé problèmes additionnels (underfitting)
   
✅ RETOUR À hidden=64:
   • Meilleur jusqu'à présent
   • 62.02% test accuracy
   • À utiliser comme baseline
   
🚀 PROCHAINE STRATÉGIE:
   • Ajouter data augmentation (pas réduire modèle)
   • Tester avec hidden=64
   • Évaluer amélioration

═══════════════════════════════════════════════════════════════════
"""

with open('RAPPORT_hidden32_ECHEC.txt', 'w', encoding='utf-8') as f:
    f.write(rapport)

print(rapport)
print("\n✅ Rapport sauvegardé: RAPPORT_hidden32_ECHEC.txt")
