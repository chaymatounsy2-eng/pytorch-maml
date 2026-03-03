from PIL import Image
import os

# Tous les chemins à vérifier
image_paths = [
    './data/background_set/meta_train/lettuce/healthy',
    './data/background_set/meta_train/lettuce/diseased',
    './data/background_set/meta_train/riz/healthy',
    './data/background_set/meta_train/riz/diseased',
    './data/background_set/meta_val/lettuce/healthy',
    './data/background_set/meta_val/lettuce/diseased',
    './data/background_set/meta_val/riz/healthy',
    './data/background_set/meta_val/riz/diseased',
    './data/evaluation_set/olive/healthy',
    './data/evaluation_set/olive/diseased',
]

print("\n" + "="*80)
print("📊 VÉRIFICATION DE TOUS LES TYPES D'IMAGES THERMIQUES")
print("="*80 + "\n")

for image_path in image_paths:
    if not os.path.exists(image_path):
        print(f"❌ {image_path} - DOSSIER N'EXISTE PAS\n")
        continue
    
    files = os.listdir(image_path)
    
    if len(files) == 0:
        print(f"⚠️  {image_path} - DOSSIER VIDE\n")
        continue
    
    print(f"✅ {image_path}")
    print(f"   📁 Total: {len(files)} images\n")
    
    # Vérifier les 3 premières images
    for i, filename in enumerate(files[:3]):
        full_path = os.path.join(image_path, filename)
        
        try:
            img = Image.open(full_path)
            width, height = img.size
            channels = len(img.getbands())
            
            channel_names = {
                1: 'Grayscale (B&W)',
                3: 'RGB (Couleur)',
                4: 'RGBA (Couleur + Alpha)'
            }
            channel_name = channel_names.get(channels, f'{channels} canaux')
            
            print(f"   Image {i+1}: {filename}")
            print(f"     📏 Taille: {width}×{height} pixels")
            print(f"     🎨 Canaux: {channels} ({channel_name})")
            print(f"     📦 Format: {img.format}")
            print()
        
        except Exception as e:
            print(f"   ❌ Erreur: {e}\n")

print("="*80)
print("✅ VÉRIFICATION TERMINÉE")
print("="*80 + "\n")