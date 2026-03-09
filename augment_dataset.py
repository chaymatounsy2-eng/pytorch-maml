"""
Script de DATA AUGMENTATION SÉPARÉ
===================================

OBJECTIF:
  Augmenter les images et les SAUVEGARDER dans data_aug/
  (Pas d'augmentation dynamique pendant training)

STRUCTURE ATTENDUE AVANT:
  data/background_set/
  ├── meta_train/
  │   ├── lettuce/healthy/ (31 originales)
  │   ├── lettuce/diseased/ (31 originales)
  │   ├── riz/healthy/ (74 originales)
  │   └── riz/diseased/ (251 originales)
  └── meta_val/
      ├── lettuce/healthy/ (8 originales)
      ├── lettuce/diseased/ (8 originales)
      ├── riz/healthy/ (93 originales)
      └── riz/diseased/ (314 originales)

STRUCTURE ATTENDUE APRÈS:
  data_aug/background_set/
  ├── meta_train/
  │   ├── lettuce/healthy/ (31 + 31×K augmentées = 31×(K+1))
  │   ├── lettuce/diseased/ (31 + 31×K augmentées = 31×(K+1))
  │   ├── riz/healthy/ (74 + 74×K augmentées = 74×(K+1))
  │   └── riz/diseased/ (251 + 251×K augmentées = 251×(K+1))
  └── meta_val/
      ├── lettuce/healthy/ (8 SANS augmentation)
      ├── lettuce/diseased/ (8 SANS augmentation)
      ├── riz/healthy/ (93 SANS augmentation)
      └── riz/diseased/ (314 SANS augmentation)

UTILISATION:
  python augment_dataset.py ./data ./data_aug --augment-factor 3
  
  (Augmente chaque image 3 fois)
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

def augment_dataset(source_root, dest_root, augment_factor=2, seed=42):
    """
    Augmenter les images et les sauvegarder
    
    Arguments:
    - source_root: Chemin vers data/ original
    - dest_root: Chemin vers data_aug/ (destination)
    - augment_factor: Nombre de versions augmentées par image (2 = image + 1 aug)
    - seed: Seed pour reproductibilité
    """
    
    random.seed(seed)
    
    print("=" * 80)
    print("DATA AUGMENTATION - SAUVEGARDE DES IMAGES AUGMENTÉES")
    print("=" * 80)
    print(f"\n📂 Source: {source_root}")
    print(f"📂 Destination: {dest_root}")
    print(f"📊 Augment factor: {augment_factor} (chaque image produit {augment_factor-1} augmentées)")
    
    # ===== TRANSFORMS POUR AUGMENTATION LÉGÈRE =====
    augmentation_pipeline = transforms.Compose([
        transforms.RandomRotation(5),                      # ±5°
        transforms.RandomHorizontalFlip(p=0.2),            # 20%
        transforms.RandomAffine(degrees=0, 
                               translate=(0.05, 0.05)),   # ±5% shift
        transforms.ColorJitter(brightness=0.1,             # ±10%
                              contrast=0.1)                # ±10%
    ])
    
    # ===== STRUCTURE DES DONNÉES =====
    background_root_src = Path(source_root) / 'background_set'
    background_root_dst = Path(dest_root) / 'background_set'
    
    species_list = ['lettuce', 'riz']
    health_states = ['healthy', 'diseased']
    splits = ['meta_train', 'meta_val']
    
    print("\n" + "=" * 80)
    print("ÉTAPE 1: CRÉATION DE LA STRUCTURE DE DESTINATION")
    print("=" * 80)
    
    # Créer la structure destination
    for split in splits:
        split_dst = background_root_dst / split
        split_dst.mkdir(parents=True, exist_ok=True)
        
        for species in species_list:
            species_dst = split_dst / species
            species_dst.mkdir(parents=True, exist_ok=True)
            
            for health_state in health_states:
                health_dst = species_dst / health_state
                health_dst.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Structure créée pour {split}/")
    
    print("\n" + "=" * 80)
    print("ÉTAPE 2: COPIE ET AUGMENTATION DES IMAGES")
    print("=" * 80)
    
    stats = {}
    
    for split in splits:
        print(f"\n📊 Traitement: {split}")
        print("-" * 80)
        
        stats[split] = {}
        
        for species in species_list:
            print(f"\n  🌾 {species}:")
            
            stats[split][species] = {}
            
            for health_state in health_states:
                # Chemins source et destination
                src_path = background_root_src / split / species / health_state
                dst_path = background_root_dst / split / species / health_state
                
                if not src_path.exists():
                    print(f"    ⚠️  {health_state}: source non trouvée")
                    stats[split][species][health_state] = 0
                    continue
                
                # Récupérer toutes les images
                image_files = [f for f in src_path.glob('*') 
                              if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']]
                
                total_images = len(image_files)
                
                if total_images == 0:
                    print(f"    ⚠️  {health_state}: aucune image")
                    stats[split][species][health_state] = 0
                    continue
                
                # ===== TRAITER CHAQUE IMAGE =====
                final_count = 0
                
                for img_file in image_files:
                    try:
                        img = Image.open(img_file).convert('RGB')
                        
                        # Copier l'image originale
                        dst_file = dst_path / img_file.name
                        img.save(str(dst_file))
                        final_count += 1
                        
                        # ===== CRÉER VERSIONS AUGMENTÉES =====
                        if split == 'meta_train':
                            # Augmenter seulement meta_train
                            for aug_idx in range(augment_factor - 1):
                                # Appliquer augmentation
                                aug_img = augmentation_pipeline(img)
                                
                                # Sauvegarder avec suffix
                                name_parts = img_file.stem.split('.')
                                aug_name = f"{name_parts[0]}_aug{aug_idx+1}{img_file.suffix}"
                                aug_file = dst_path / aug_name
                                aug_img.save(str(aug_file))
                                final_count += 1
                    
                    except Exception as e:
                        print(f"    ❌ Erreur: {img_file.name}: {e}")
                        continue
                
                print(f"    📋 {health_state}:")
                print(f"       Originales: {total_images}")
                print(f"       Augmentées: {final_count - total_images}")
                print(f"       Total: {final_count}")
                
                stats[split][species][health_state] = final_count
    
    # ===== AFFICHER STATISTIQUES FINALES =====
    print("\n" + "=" * 80)
    print("ÉTAPE 3: RÉSUMÉ FINAL")
    print("=" * 80)
    
    for split in splits:
        print(f"\n📊 {split.upper()}:")
        total_split = 0
        
        for species in species_list:
            print(f"  🌾 {species}:")
            total_species = 0
            
            for health_state in health_states:
                count = stats[split][species].get(health_state, 0)
                total_species += count
                
                if count > 0:
                    if split == 'meta_train':
                        print(f"     • {health_state}: {count} images")
                    else:
                        print(f"     • {health_state}: {count} images (pas augmentation)")
            
            print(f"     → Total {species}: {total_species} images")
            total_split += total_species
        
        print(f"  → Total {split}: {total_split} images")
    
    print("\n" + "=" * 80)
    print("✅ AUGMENTATION COMPLÈTE!")
    print("=" * 80)
    print(f"\n📂 Toutes les images sont dans: {dest_root}/")
    print(f"\n✨ Vous pouvez maintenant entraîner avec:")
    print(f"   python train.py {dest_root} --dataset thermal ...")
    
    return True

if __name__ == '__main__':
    import sys
    
    # Paramètres par défaut
    source_root = './data'
    dest_root = './data_aug'
    augment_factor = 2  # 1 original + 1 augmentée = 2 total
    
    # Récupérer depuis ligne de commande
    if len(sys.argv) > 1:
        source_root = sys.argv[1]
    if len(sys.argv) > 2:
        dest_root = sys.argv[2]
    if len(sys.argv) > 3:
        augment_factor = int(sys.argv[3])
    
    print(f"\n🚀 Démarrage augmentation...")
    print(f"   Source: {source_root}")
    print(f"   Destination: {dest_root}")
    print(f"   Augment factor: {augment_factor}\n")
    
    success = augment_dataset(source_root, dest_root, augment_factor, seed=42)
    
    if success:
        print("\n✨ Augmentation réussie!")
        sys.exit(0)
    else:
        print("\n❌ Erreur")
        sys.exit(1)