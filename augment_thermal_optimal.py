"""
Augmentation OPTIMALE pour Images Thermiques
==============================================

OPTIMISÉE POUR:
  • Images thermiques FLIR (RGB)
  • Patterns subtils
  • Minimal artifacting

AUGMENTATIONS LÉGÈRES:
  ✅ RandomRotation(2°) - très léger
  ✅ RandomHorizontalFlip(10%) - rare
  ✅ SANS ColorJitter - brouille patterns thermiques
  ✅ SANS RandomAffine - déforme textures
  
STRUCTURE:
  data_aug/ (original)
  └── data_aug_thermal_opt/ (NEW - avec augmentation optimale)
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

def augment_thermal_optimal(source_root, dest_root, augment_factor=3, seed=42):
    """
    Augmentation OPTIMALE pour images thermiques
    
    Arguments:
    - source_root: Chemin vers data_aug/
    - dest_root: Chemin vers data_aug_thermal_opt/ (destination)
    - augment_factor: Nombre de versions par image (3 = original + 2 aug)
    - seed: Pour reproductibilité
    """
    
    random.seed(seed)
    
    print("=" * 80)
    print("AUGMENTATION OPTIMALE POUR IMAGES THERMIQUES")
    print("=" * 80)
    print(f"\n📂 Source: {source_root}")
    print(f"📂 Destination: {dest_root}")
    print(f"📊 Augment factor: {augment_factor}")
    print(f"   (original + {augment_factor-1} augmentées)")
    
    # ===== AUGMENTATION LÉGÈRE POUR THERMIQUES =====
    augmentation_pipeline = transforms.Compose([
        # ✅ Rotation très légère (±2° au lieu de ±5°)
        transforms.RandomRotation(2),
        
        # ✅ Flip horizontal rare (10% au lieu de 20%)
        transforms.RandomHorizontalFlip(p=0.1),
        
        # ❌ PAS ColorJitter (brouille patterns thermiques)
        # ❌ PAS RandomAffine (déforme textures importantes)
        # ❌ PAS RandomVerticalFlip (peut perdre infos)
    ])
    
    print("\n" + "=" * 80)
    print("AUGMENTATIONS APPLIQUÉES:")
    print("=" * 80)
    print(f"  ✅ RandomRotation: ±2° (très léger)")
    print(f"  ✅ RandomHorizontalFlip: 10% (rare)")
    print(f"  ❌ PAS ColorJitter (préserve informations thermiques)")
    print(f"  ❌ PAS RandomAffine (préserve géométrie)")
    
    # ===== STRUCTURE DES DONNÉES =====
    background_root_src = Path(source_root) / 'background_set'
    background_root_dst = Path(dest_root) / 'background_set'
    
    species_list = ['lettuce', 'riz']
    health_states = ['healthy', 'diseased']
    splits = ['meta_train', 'meta_val']
    
    print("\n" + "=" * 80)
    print("ÉTAPE 1: CRÉATION DE LA STRUCTURE")
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
    
    # Copier aussi evaluation_set
    eval_src = Path(source_root) / 'evaluation_set'
    eval_dst = Path(dest_root) / 'evaluation_set'
    if eval_src.exists():
        for species in ['olive']:
            (eval_dst / species / 'healthy').mkdir(parents=True, exist_ok=True)
            (eval_dst / species / 'diseased').mkdir(parents=True, exist_ok=True)
        print(f"✅ Structure créée pour evaluation_set/")
    
    print("\n" + "=" * 80)
    print("ÉTAPE 2: COPIE ET AUGMENTATION")
    print("=" * 80)
    
    stats = {}
    total_images_created = 0
    
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
                        
                        # ✅ Copier l'image originale
                        dst_file = dst_path / img_file.name
                        img.save(str(dst_file))
                        final_count += 1
                        total_images_created += 1
                        
                        # ===== CRÉER VERSIONS AUGMENTÉES =====
                        # SEULEMENT pour meta_train (pas meta_val!)
                        if split == 'meta_train':
                            for aug_idx in range(augment_factor - 1):
                                # Appliquer augmentation
                                aug_img = augmentation_pipeline(img)
                                
                                # Sauvegarder avec suffix
                                name_parts = img_file.stem.split('.')
                                aug_name = f"{name_parts[0]}_aug{aug_idx+1}{img_file.suffix}"
                                aug_file = dst_path / aug_name
                                aug_img.save(str(aug_file))
                                final_count += 1
                                total_images_created += 1
                    
                    except Exception as e:
                        print(f"    ❌ Erreur: {img_file.name}: {e}")
                        continue
                
                print(f"    📋 {health_state}:")
                print(f"       Originales: {total_images}")
                
                if split == 'meta_train':
                    print(f"       Augmentées: {final_count - total_images}")
                else:
                    print(f"       (meta_val: PAS d'augmentation)")
                
                print(f"       Total: {final_count}")
                
                stats[split][species][health_state] = final_count
    
    # ===== COPIER evaluation_set (SANS augmentation) =====
    print(f"\n📊 Traitement: evaluation_set")
    print("-" * 80)
    
    eval_src = Path(source_root) / 'evaluation_set'
    if eval_src.exists():
        for species in ['olive']:
            print(f"\n  🌾 {species}:")
            
            for health_state in health_states:
                src_eval = eval_src / species / health_state
                dst_eval = Path(dest_root) / 'evaluation_set' / species / health_state
                
                if src_eval.exists():
                    image_files = [f for f in src_eval.glob('*') 
                                  if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']]
                    
                    for img_file in image_files:
                        try:
                            shutil.copy2(str(img_file), str(dst_eval / img_file.name))
                        except:
                            pass
                    
                    print(f"    ✅ {health_state}: {len(image_files)} images (SANS augmentation)")
    
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
                        print(f"     • {health_state}: {count} images (SANS aug)")
            
            print(f"     → Total {species}: {total_species} images")
            total_split += total_species
        
        print(f"  → Total {split}: {total_split} images")
    
    print("\n" + "=" * 80)
    print("📈 RÉSUMÉ GLOBAL")
    print("=" * 80)
    print(f"\n✅ Total images créées: {total_images_created} images")
    print(f"✅ Destination: {dest_root}/")
    
    print("\n" + "=" * 80)
    print("✨ AUGMENTATION COMPLÈTE!")
    print("=" * 80)
    print(f"\n📂 Nouvelle structure créée dans: {dest_root}/")
    print(f"\n✅ Prêt pour entraînement!")
    print(f"\nCommande pour entraîner:")
    print(f"   python train.py {dest_root} --dataset thermal --num-ways 2 --num-shots 5 --num-shots-test 10 --batch-size 2 --num-steps 1 --step-size 0.01 --meta-lr 0.001 --num-epochs 150 --num-batches 100 --hidden-size 64 --output-folder ./results_thermal_opt --verbose")
    
    return True

if __name__ == '__main__':
    import sys
    
    # Paramètres par défaut
    source_root = './data_aug'
    dest_root = './data_aug_thermal_opt'
    augment_factor = 3  # original + 2 augmentées
    
    # Récupérer depuis ligne de commande
    if len(sys.argv) > 1:
        source_root = sys.argv[1]
    if len(sys.argv) > 2:
        dest_root = sys.argv[2]
    if len(sys.argv) > 3:
        augment_factor = int(sys.argv[3])
    
    print(f"\n🚀 Démarrage augmentation optimale pour thermiques...")
    print(f"   Source: {source_root}")
    print(f"   Destination: {dest_root}")
    print(f"   Augment factor: {augment_factor}\n")
    
    success = augment_thermal_optimal(source_root, dest_root, augment_factor, seed=42)
    
    if success:
        print("\n✨ Augmentation thermique réussie!")
        sys.exit(0)
    else:
        print("\n❌ Erreur")
        sys.exit(1)