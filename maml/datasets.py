import torch.nn.functional as F
import os
import random
from pathlib import Path
from collections import namedtuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from maml.utils import ToTensor1D

# ============================================================================
# BENCHMARK NAMEDTUPLE
# ============================================================================
Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')

# ============================================================================
# TRANSFORMATION POUR IMAGES THERMIQUES (RGB)
# ============================================================================
THERMAL_MEAN = [0.485, 0.456, 0.406]
THERMAL_STD = [0.229, 0.224, 0.225]

THERMAL_TRANSFORM = Compose([
    Resize((84, 84)),  # ✅ CHANGÉ de 224 à 84 (taille originale MiniImagenet)
    ToTensor(),
    Normalize(mean=THERMAL_MEAN, std=THERMAL_STD)
])

# ============================================================================
# CLASSE THERMALMETADATASET
# ============================================================================
class ThermalMetaDataset(Dataset):
    """
    Dataset MAML pour images thermiques FLIR
    Structure: root/species/healthy/*.jpg et root/species/diseased/*.jpg
    """
    
    def __init__(self, root, species_list, num_shots, num_shots_test, num_tasks=100000):
        self.root = root
        self.species_list = species_list
        self.num_shots = num_shots
        self.num_shots_test = num_shots_test
        self.num_tasks = num_tasks
        
        self.data = {}
        
        print(f"\n[ThermalMetaDataset] Chargement depuis: {root}")
        print(f"[ThermalMetaDataset] Espèces: {species_list}")
        
        for species in species_list:
            self.data[species] = {}
            
            # Dossier healthy
            healthy_dir = os.path.join(root, species, 'healthy')
            if os.path.exists(healthy_dir):
                healthy_files = [os.path.join(healthy_dir, f) 
                                for f in os.listdir(healthy_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                self.data[species][0] = healthy_files
                print(f"  ✅ {species}/healthy: {len(healthy_files)} images")
            else:
                self.data[species][0] = []
                print(f"  ⚠️  {species}/healthy: dossier non trouvé")
            
            # Dossier diseased
            diseased_dir = os.path.join(root, species, 'diseased')
            if os.path.exists(diseased_dir):
                diseased_files = [os.path.join(diseased_dir, f) 
                                 for f in os.listdir(diseased_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                self.data[species][1] = diseased_files
                print(f"  ✅ {species}/diseased: {len(diseased_files)} images")
            else:
                self.data[species][1] = []
                print(f"  ⚠️  {species}/diseased: dossier non trouvé")
    
    def __len__(self):
        return self.num_tasks
    
    def _load_images(self, paths):
        """Charger et transformer les images"""
        images = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                img = THERMAL_TRANSFORM(img)
                images.append(img)
            except Exception as e:
                print(f"  ❌ Erreur lors du chargement de {p}: {e}")
                continue
        
        if len(images) == 0:
            raise RuntimeError(f"Aucune image chargée parmi {len(paths)} chemins")
        
        return torch.stack(images)
    
    def __getitem__(self, idx):
        """Générer une tâche MAML"""
        species = random.choice(self.species_list)
        
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        for label in [0, 1]:
            total_needed = self.num_shots + self.num_shots_test
            available_paths = self.data[species].get(label, [])
            
            if len(available_paths) < total_needed:
                samples = random.choices(available_paths, k=total_needed)
            else:
                samples = random.sample(available_paths, k=total_needed)
            
            support_samples = samples[:self.num_shots]
            query_samples = samples[self.num_shots:]
            
            support_x.append(self._load_images(support_samples))
            query_x.append(self._load_images(query_samples))
            
            support_y += [label] * self.num_shots
            query_y += [label] * self.num_shots_test
        
        support_x = torch.cat(support_x, dim=0)
        query_x = torch.cat(query_x, dim=0)
        
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_y = torch.tensor(query_y, dtype=torch.long)
        
        return {
            'train': (support_x, support_y),
            'test': (query_x, query_y)
        }

# ============================================================================
# FONCTION PRINCIPALE GET_BENCHMARK_BY_NAME
# ============================================================================
def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None):
    """
    Fonction factory pour créer des benchmarks MAML
    """
    
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    
    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'thermal':
        """
        Support pour images thermiques FLIR (RGB)
        
        Structure attendue:
        folder/
        ├── background_set/
        │   ├── meta_train/
        │   │   ├── lettuce/
        │   │   │   ├── healthy/
        │   │   │   └── diseased/
        │   │   └── riz/
        │   │       ├── healthy/
        │   │       └── diseased/
        │   └── meta_val/
        │       ├── lettuce/
        │       │   ├── healthy/
        │       │   └── diseased/
        │       └── riz/
        │           ├── healthy/
        │           └── diseased/
        └── evaluation_set/
            └── olive/
                ├── healthy/
                └── diseased/
        """
        
        background_root = os.path.join(folder, 'background_set')
        meta_train_path = os.path.join(background_root, 'meta_train')
        meta_val_path = os.path.join(background_root, 'meta_val')
        evaluation_root = os.path.join(folder, 'evaluation_set')
        
        print("\n" + "="*80)
        print("CHARGEMENT DU DATASET THERMAL (IMAGES FLIR - RGB)")
        print("="*80)
        
        meta_train_dataset = ThermalMetaDataset(
            root=meta_train_path,
            species_list=['lettuce', 'riz'],
            num_shots=num_shots,
            num_shots_test=num_shots_test
        )
        
        meta_val_dataset = ThermalMetaDataset(
            root=meta_val_path,
            species_list=['lettuce', 'riz'],
            num_shots=num_shots,
            num_shots_test=num_shots_test
        )
        
        meta_test_dataset = ThermalMetaDataset(
            root=evaluation_root,
            species_list=['olive'],
            num_shots=num_shots,
            num_shots_test=num_shots_test
        )
        
        # Utiliser ModelConvMiniImagenet pour RGB (3 canaux, 224×224)
        model = ModelConvMiniImagenet(
            out_features=num_ways,
            hidden_size=hidden_size if hidden_size is not None else 64
        )
        
        loss_function = F.cross_entropy
        
        print("="*80 + "\n")

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)