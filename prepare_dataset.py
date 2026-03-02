"""
Script de préparation du dataset pour MAML
=============================================

OBJECTIF:
Diviser automatiquement les images en meta_train (80%) et meta_val (20%)

STRUCTURE ATTENDUE AVANT:
data/background_set/
├── meta_train/
│   ├── lettuce/
│   │   ├── healthy/ (TOUTES les images ici)
│   │   └── diseased/ (TOUTES les images ici)
│   ��── riz/
│       ├── healthy/ (TOUTES les images ici)
│       └── diseased/ (TOUTES les images ici)
└── meta_val/ (VIDE)

STRUCTURE ATTENDUE APRÈS:
data/background_set/
├── meta_train/
│   ├── lettuce/
│   │   ├── healthy/ (80% des images)
│   │   └── diseased/ (80% des images)
│   └── riz/
│       ├── healthy/ (80% des images)
│       └── diseased/ (80% des images)
└── meta_val/
    ├── lettuce/
    │   ├── healthy/ (20% des images)
    │   └── diseased/ (20% des images)
    └── riz/
        ├── healthy/ (20% des images)
        └── diseased/ (20% des images)

POURQUOI CE SPLIT?
- meta_train: Utilisé pour l'entraînement MAML (train.py ligne 72)
- meta_val: Utilisé pour la validation pendant l'entraînement (train.py ligne 75)
  → Cela permet de vérifier que le modèle généralise bien sur d'autres tâches
"""

import os
import shutil
import random
from pathlib import Path

def split_dataset(data_root, train_ratio=0.8):
    """
    ÉTAPE 1: Diviser les images entre meta_train et meta_val
    
    Arguments:
    - data_root: Chemin vers 'data/background_set'
    - train_ratio: 0.8 = 80% train, 20% val
    
    Retour:
    - Affiche les statistiques du split
    """
    
    background_root = Path(data_root) / 'background_set'
    meta_train_root = background_root / 'meta_train'
    meta_val_root = background_root / 'meta_val'
    
    print("=" * 80)
    print("PRÉPARATION DU DATASET POUR MAML")
    print("=" * 80)
    print(f"\n📂 Chemin racine: {data_root}")
    print(f"📂 Background set: {background_root}")
    
    # ÉTAPE 1: Vérifier que meta_train existe
    if not meta_train_root.exists():
        print(f"\n❌ ERREUR: {meta_train_root} n'existe pas!")
        return False
    
    print(f"\n✅ Dossier meta_train trouvé: {meta_train_root}")
    
    # ÉTAPE 2: Créer meta_val s'il n'existe pas
    meta_val_root.mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossier meta_val prêt: {meta_val_root}")
    
    # ÉTAPE 3: Scanner toutes les classes (lettuce, riz)
    print("\n" + "=" * 80)
    print("ÉTAPE 1: SCAN DES CLASSES")
    print("=" * 80)
    
    species_list = ['lettuce', 'riz']
    health_states = ['healthy', 'diseased']
    
    for species in species_list:
        species_train_dir = meta_train_root / species
        
        if not species_train_dir.exists():
            print(f"\n⚠️  {species} n'existe pas dans meta_train, création...")
            species_train_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"\n✅ Classe trouvée: {species}")
    
    # ÉTAPE 4: Pour chaque espèce et état de santé, diviser les images
    print("\n" + "=" * 80)
    print("ÉTAPE 2: DIVISION DES IMAGES (80% train / 20% val)")
    print("=" * 80)
    
    total_stats = {species: {state: {'total': 0, 'train': 0, 'val': 0} 
                             for state in health_states} 
                   for species in species_list}
    
    for species in species_list:
        print(f"\n📊 Traitement: {species}")
        print("-" * 80)
        
        for health_state in health_states:
            # CHEMINS SOURCE ET DESTINATION
            source_dir = meta_train_root / species / health_state
            train_dest = meta_train_root / species / health_state
            val_dest = meta_val_root / species / health_state
            
            # Créer meta_val/species/health_state s'il n'existe pas
            val_dest.mkdir(parents=True, exist_ok=True)
            
            # ÉTAPE 5: Scanner tous les images
            if not source_dir.exists():
                print(f"  ⚠️  {health_state}: dossier inexistant")
                continue
            
            # Récupérer tous les images (.jpg, .png, etc.)
            image_files = [f for f in source_dir.glob('*') 
                          if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']]
            
            total_images = len(image_files)
            
            if total_images == 0:
                print(f"  ⚠️  {health_state}: aucune image trouvée")
                continue
            
            # ÉTAPE 6: Calculer le split 80/20
            num_train = int(total_images * train_ratio)
            num_val = total_images - num_train
            
            # Mélanger aléatoirement les images
            random.shuffle(image_files)
            
            # Diviser
            train_images = image_files[:num_train]
            val_images = image_files[num_train:]
            
            # ÉTAPE 7: Copier les images de validation vers meta_val
            print(f"  📋 {health_state}:")
            print(f"     Total: {total_images} images")
            print(f"     → Train (80%): {len(train_images)} images → {train_dest}")
            print(f"     → Val (20%):   {len(val_images)} images → {val_dest}")
            
            for img_file in val_images:
                try:
                    dest_path = val_dest / img_file.name
                    shutil.move(str(img_file), str(dest_path))
                except Exception as e:
                    print(f"     ❌ Erreur lors du déplacement {img_file.name}: {e}")
            
            # Mettre à jour les statistiques
            total_stats[species][health_state]['total'] = total_images
            total_stats[species][health_state]['train'] = len(train_images)
            total_stats[species][health_state]['val'] = len(val_images)
    
    # ÉTAPE 8: Afficher les statistiques finales
    print("\n" + "=" * 80)
    print("ÉTAPE 3: RÉSUMÉ FINAL DU SPLIT")
    print("=" * 80)
    
    print("\n📊 STATISTIQUES PAR CLASSE:")
    print("-" * 80)
    
    total_train_count = 0
    total_val_count = 0
    
    for species in species_list:
        print(f"\n🌾 {species.upper()}:")
        
        for health_state in health_states:
            stats = total_stats[species][health_state]
            if stats['total'] > 0:
                print(f"   {health_state}:")
                print(f"     • Total: {stats['total']} images")
                print(f"     • Meta-train: {stats['train']} images ({stats['train']/stats['total']*100:.1f}%)")
                print(f"     • Meta-val: {stats['val']} images ({stats['val']/stats['total']*100:.1f}%)")
                
                total_train_count += stats['train']
                total_val_count += stats['val']
    
    print("\n" + "=" * 80)
    print("📈 TOTAUX FINAUX:")
    print("-" * 80)
    print(f"✅ Meta-train (80%): {total_train_count} images")
    print(f"✅ Meta-val (20%):   {total_val_count} images")
    print(f"✅ TOTAL:            {total_train_count + total_val_count} images")
    
    # ÉTAPE 9: Vérifier la structure finale
    print("\n" + "=" * 80)
    print("ÉTAPE 4: VÉRIFICATION DE LA STRUCTURE")
    print("=" * 80)
    
    print("\n📂 Arborescence finale:")
    print("\ndata/background_set/")
    
    for split in ['meta_train', 'meta_val']:
        split_dir = background_root / split
        print(f"├── {split}/")
        
        for species in species_list:
            species_dir = split_dir / species
            print(f"│   ├── {species}/")
            
            for health_state in health_states:
                health_dir = species_dir / health_state
                if health_dir.exists():
                    num_images = len(list(health_dir.glob('*')))
                    print(f"│   │   ├── {health_state}/ ({num_images} images)")
                else:
                    print(f"│   │   ├── {health_state}/ (vide)")
    
    print("\n✅ PRÉPARATION COMPLÈTE!")
    print("\nVous pouvez maintenant lancer l'entraînement avec:")
    print("python train.py ./data --dataset thermal --num-ways 2 --num-shots 5 ...")
    
    return True

if __name__ == '__main__':
    import sys
    
    # Récupérer le chemin racine depuis la ligne de commande
    # ou utiliser le répertoire courant
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = './data'
    
    print(f"\n🚀 Démarrage de la préparation du dataset...")
    print(f"   Chemin: {data_root}\n")
    
    success = split_dataset(data_root, train_ratio=0.8)
    
    if success:
        print("\n✨ Dataset prêt pour MAML!")
        sys.exit(0)
    else:
        print("\n❌ Erreur lors de la préparation du dataset")
        sys.exit(1)