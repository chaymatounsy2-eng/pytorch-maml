import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

def apply_data_augmentation(base_dir, augmentation_factor=2):
    """
    Applique de la data augmentation sur les images du meta_train avec Albumentations.
    
    Args:
        base_dir: Chemin vers data_aug/background_set
        augmentation_factor: Nombre de versions augmentées par image (défaut: 2)
    """
    
    base_dir = os.path.abspath(base_dir)
    print(f"📍 Chemin de base: {base_dir}\n")
    
    # Définir les augmentations (compatible avec NumPy 2.0+)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # Flip horizontal 50%
        A.Rotate(limit=25, p=0.7),  # Rotation aléatoire ±25°
        A.RandomScale(scale_limit=0.2, p=0.7),  # Zoom aléatoire
        A.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, p=0.7),  # Translation
        A.GaussNoise(p=0.5),  # Bruit gaussien
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),  # Luminosité & contraste
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.75, 1.5), p=0.5),  # Sharpening
        A.GaussianBlur(blur_limit=3, p=0.3),  # Flou gaussien léger
        A.Perspective(scale=(0.05, 0.1), p=0.3),  # Perspective
    ], p=1.0)
    
    crops = ['lettuce', 'riz']
    statuses = ['healthy', 'diseased']
    
    print("🔄 Application de la data augmentation...\n")
    
    total_original = 0
    total_augmented = 0
    
    for crop in crops:
        for status in statuses:
            train_path = os.path.join(base_dir, 'meta_train', crop, status)
            
            if not os.path.exists(train_path):
                print(f"⚠️  Chemin introuvable: {train_path}")
                continue
            
            # Récupérer toutes les images
            image_files = [f for f in os.listdir(train_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) 
                          and not f.startswith('.')]
            
            if not image_files:
                print(f"⚠️  Aucune image trouvée dans {crop}/{status}")
                continue
            
            print(f"✅ {crop.upper()} - {status.upper()}")
            print(f"   Images originales: {len(image_files)}")
            
            augmented_count = 0
            
            for img_name in image_files:
                img_path = os.path.join(train_path, img_name)
                
                try:
                    # Lire l'image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"   ⚠️  Erreur lecture: {img_name}")
                        continue
                    
                    # Convertir BGR vers RGB pour Albumentations
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Générer des versions augmentées
                    for i in range(augmentation_factor):
                        # Appliquer augmentation
                        augmented = transform(image=image_rgb)
                        augmented_image = augmented['image']
                        
                        # Reconvertir en BGR pour cv2.imwrite
                        augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                        
                        # Créer un nom unique pour l'image augmentée
                        name, ext = os.path.splitext(img_name)
                        augmented_name = f"{name}_aug_{i}{ext}"
                        augmented_path = os.path.join(train_path, augmented_name)
                        
                        # Sauvegarder l'image augmentée
                        cv2.imwrite(augmented_path, augmented_image_bgr)
                        augmented_count += 1
                
                except Exception as e:
                    print(f"   ⚠️  Erreur traitement {img_name}: {e}")
            
            total_original += len(image_files)
            total_augmented += augmented_count
            
            final_count = len([f for f in os.listdir(train_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                             and not f.startswith('.')])
            print(f"   Images augmentées: {augmented_count}")
            print(f"   Total dans le dossier: {final_count}\n")
    
    print("="*60)
    print(f"📊 RÉSUMÉ FINAL DE L'AUGMENTATION\n")
    print(f"Images originales: {total_original}")
    print(f"Images augmentées: {total_augmented}")
    print(f"Total: {total_original + total_augmented}")
    print("="*60 + "\n")
    
    print_train_summary(base_dir)


def print_train_summary(base_dir):
    """Affiche un résumé du meta_train après augmentation"""
    
    crops = ['lettuce', 'riz']
    statuses = ['healthy', 'diseased']
    
    print("🟢 RÉSUMÉ meta_train APRÈS AUGMENTATION:\n")
    
    grand_total = 0
    for crop in crops:
        print(f"{crop.upper()}:")
        crop_total = 0
        for status in statuses:
            path = os.path.join(base_dir, 'meta_train', crop, status)
            count = len([f for f in os.listdir(path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                       and not f.startswith('.')]) if os.path.exists(path) else 0
            crop_total += count
            print(f"  └─ {status}: {count} images")
        
        print(f"  Total {crop}: {crop_total}\n")
        grand_total += crop_total
    
    print(f"{'='*60}")
    print(f"🎯 TOTAL meta_train: {grand_total} images")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    BASE_DIR = r"C:\Users\user\Desktop\pytorch-maml-v2\data_aug\background_set"
    
    # augmentation_factor=2 signifie 2 versions augmentées par image
    apply_data_augmentation(BASE_DIR, augmentation_factor=2)