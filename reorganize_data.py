import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def reorganize_existing_structure(base_dir, train_ratio=0.8, random_seed=42):
    """
    Réorganise les images existantes dans meta_train et meta_val
    en respectant le ratio 80/20 sans duplication.
    """
    
    random.seed(random_seed)
    
    # Convertir en chemin absolu
    base_dir = os.path.abspath(base_dir)
    print(f"📍 Chemin de base: {base_dir}\n")
    
    # Vérifier si le dossier existe
    if not os.path.exists(base_dir):
        print(f"❌ Dossier introuvable: {base_dir}")
        return
    
    crops = ['lettuce', 'riz']
    statuses = ['healthy', 'diseased']
    
    # D'abord, on récupère TOUTES les images de tous les dossiers
    all_images = defaultdict(lambda: defaultdict(list))
    
    print("📂 Recherche des images existantes...\n")
    
    for crop in crops:
        for status in statuses:
            # Vérifier meta_train
            train_path = os.path.join(base_dir, 'meta_train', crop, status)
            print(f"Vérification: {train_path}")
            if os.path.exists(train_path):
                images = [f for f in os.listdir(train_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                if images:
                    print(f"  ✓ Trouvé {len(images)} images")
                    all_images[crop][status].extend([(f, train_path) for f in images])
            
            # Vérifier meta_val
            val_path = os.path.join(base_dir, 'meta_val', crop, status)
            print(f"Vérification: {val_path}")
            if os.path.exists(val_path):
                images = [f for f in os.listdir(val_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                if images:
                    print(f"  ✓ Trouvé {len(images)} images")
                    all_images[crop][status].extend([(f, val_path) for f in images])
    
    # Afficher les images trouvées
    print("\n" + "="*60)
    total_images = 0
    for crop in crops:
        for status in statuses:
            count = len(all_images[crop][status])
            if count > 0:
                print(f"  {crop}/{status}: {count} images")
                total_images += count
    
    if total_images == 0:
        print("❌ Aucune image trouvée!")
        print("\nVérifiez que:")
        print("  1. Le dossier data_aug/background_set existe")
        print("  2. Les sous-dossiers meta_train et meta_val existent")
        print("  3. Les dossiers lettuce et riz existent")
        print("  4. Les dossiers healthy et diseased contiennent des images")
        return
    
    print(f"\n📊 Total: {total_images} images")
    print("="*60 + "\n")
    
    # Créer un dossier temporaire pour stocker les images
    temp_dir = os.path.join(base_dir, 'temp_images')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("🔄 Copie des images dans un dossier temporaire...\n")
    
    # Copier toutes les images dans le dossier temporaire
    for crop in crops:
        for status in statuses:
            images_info = all_images[crop][status]
            
            if not images_info:
                continue
            
            temp_crop_dir = os.path.join(temp_dir, crop, status)
            os.makedirs(temp_crop_dir, exist_ok=True)
            
            for img_name, original_path in images_info:
                src = os.path.join(original_path, img_name)
                dst = os.path.join(temp_crop_dir, img_name)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
    
    print("🔄 Réorganisation en cours...\n")
    
    # Réorganiser à partir du dossier temporaire
    for crop in crops:
        for status in statuses:
            temp_crop_status_dir = os.path.join(temp_dir, crop, status)
            
            if not os.path.exists(temp_crop_status_dir):
                continue
            
            # Récupérer les images du dossier temporaire
            images = [f for f in os.listdir(temp_crop_status_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            if not images:
                continue
            
            # Mélanger les images
            random.shuffle(images)
            
            # Diviser 80/20
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Créer les dossiers de destination
            train_path = os.path.join(base_dir, 'meta_train', crop, status)
            val_path = os.path.join(base_dir, 'meta_val', crop, status)
            
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)
            
            # Vider les dossiers existants
            for f in os.listdir(train_path):
                file_path = os.path.join(train_path, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            for f in os.listdir(val_path):
                file_path = os.path.join(val_path, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Copier les images en train
            for img_name in train_images:
                src = os.path.join(temp_crop_status_dir, img_name)
                dst = os.path.join(train_path, img_name)
                shutil.copy2(src, dst)
            
            # Copier les images en val
            for img_name in val_images:
                src = os.path.join(temp_crop_status_dir, img_name)
                dst = os.path.join(val_path, img_name)
                shutil.copy2(src, dst)
            
            # Statistiques
            print(f"✅ {crop.upper()} - {status.upper()}")
            print(f"   Train: {len(train_images)} images ({len(train_images)/len(images)*100:.1f}%)")
            print(f"   Val:   {len(val_images)} images ({len(val_images)/len(images)*100:.1f}%)\n")
    
    # Supprimer le dossier temporaire
    shutil.rmtree(temp_dir)
    
    print_final_summary(base_dir)


def print_final_summary(base_dir):
    """Affiche le résumé final"""
    
    crops = ['lettuce', 'riz']
    statuses = ['healthy', 'diseased']
    
    total_train = 0
    total_val = 0
    
    print("="*60)
    print("📊 RÉSUMÉ FINAL DE LA RÉORGANISATION\n")
    
    for meta, meta_label in [('meta_train', '🟢 TRAIN (80%)'), ('meta_val', '🔵 VALIDATION (20%)')]:
        print(f"\n{meta_label}:")
        meta_total = 0
        
        for crop in crops:
            print(f"  └─ {crop}:")
            crop_total = 0
            for status in statuses:
                path = os.path.join(base_dir, meta, crop, status)
                count = len([f for f in os.listdir(path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]) if os.path.exists(path) else 0
                crop_total += count
                print(f"     ├─ {status}: {count} images")
            
            meta_total += crop_total
        
        if meta == 'meta_train':
            total_train = meta_total
        else:
            total_val = meta_total
    
    grand_total = total_train + total_val
    
    print(f"\n{'='*60}")
    if grand_total > 0:
        print(f"📈 Total TRAIN: {total_train} images ({total_train/grand_total*100:.1f}%)")
        print(f"📉 Total VAL:  {total_val} images ({total_val/grand_total*100:.1f}%)")
        print(f"🎯 TOTAL:      {grand_total} images")
    else:
        print(f"❌ Aucune image trouvée après réorganisation!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Option 1: Chemin absolu complet
    BASE_DIR = r"C:\Users\user\Desktop\pytorch-maml-v2\data_aug\background_set"
    
    # Option 2: Si vous préférez un chemin relatif, utilisez:
    # BASE_DIR = os.path.join(os.path.dirname(__file__), 'data_aug', 'background_set')
    
    # Lancer la réorganisation
    reorganize_existing_structure(BASE_DIR, train_ratio=0.8)