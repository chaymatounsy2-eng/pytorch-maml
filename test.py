import torch
import os
import json
from torch.utils.data import DataLoader
from torchmeta.utils import gradient_update_parameters

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning
# ===== AJOUTER CES IMPORTS =====
import random
import numpy as np
# ===== FIN IMPORTS =====

# ===== SET RANDOM SEEDS FOR REPRODUCIBILITY =====
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"\n🌱 Random seed set to {SEED} for reproducibility\n")
# ===== END RANDOM SEEDS =====


def main(args):
    # ===== CHARGER LA CONFIGURATION =====
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.folder is not None:
        config['folder'] = args.folder
    if args.num_steps > 0:
        config['num_steps'] = args.num_steps
    if args.num_batches > 0:
        config['num_batches'] = args.num_batches
    
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"🧪 MAML TESTING STARTED")
    print(f"{'='*80}")
    print(f"Dataset: {config['dataset']}")
    print(f"Device: {device}")
    print(f"Model path: {config['model_path']}")
    print(f"{'='*80}\n")

    # ===== CHARGER LE BENCHMARK =====
    print("📦 Loading benchmark...")
    benchmark = get_benchmark_by_name(config['dataset'],
                                      config['folder'],
                                      config['num_ways'],
                                      config['num_shots'],
                                      config['num_shots_test'],
                                      hidden_size=config['hidden_size'])
    print("✅ Benchmark loaded!\n")

    # ===== CHARGER LE MODÈLE ENTRAÎNÉ =====
        # ===== CHARGER LE MODÈLE ENTRAÎNÉ =====
    print(f"📂 Loading model from: {config['model_path']}")
    if not os.path.exists(config['model_path']):
        print(f"❌ ERROR: Model file not found at {config['model_path']}")
        return
    
    benchmark.model.load_state_dict(torch.load(config['model_path'], map_location=device))
    benchmark.model.to(device)  # ✅ CHANGÉ: Sans assignation
    benchmark.model.eval()  # Mode évaluation
    print("✅ Model loaded!\n")

    # ===== CHOISIR LE TYPE DE DATALOADER =====
    if config['dataset'] == 'thermal':
        print("🔥 THERMAL DATASET - Using custom test loop\n")
        
        # Utiliser DataLoader simple pour thermal
        print("📦 Creating test dataloader...")
        meta_test_dataloader = DataLoader(benchmark.meta_test_dataset,
                                          batch_size=config['batch_size'],
                                          shuffle=False,
                                          num_workers=0)
        print("✅ Test dataloader created!\n")

        # ===== BOUCLE DE TEST PERSONNALISÉE POUR THERMAL =====
        print(f"{'='*80}")
        print(f"🧪 TESTING ON OLIVE (Unseen Species)")
        print(f"{'='*80}\n")
        
        test_loss = 0.0
        test_accuracy = 0.0
        test_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(meta_test_dataloader):
                if test_count >= config['num_batches']:
                    print(f"   ✅ Max test batches reached ({config['num_batches']})")
                    break
                
                try:
                    # Extraire support et query sets
                    train_x, train_y = batch['train']
                    test_x, test_y = batch['test']
                    
                    # Reshape et envoyer sur GPU
                    train_x = train_x.view(-1, 3, 84, 84).to(device)
                    train_y = train_y.view(-1).to(device)
                    test_x = test_x.view(-1, 3, 84, 84).to(device)
                    test_y = test_y.view(-1).to(device)
                    
                    # ===== INNER LOOP: Adaptation - BESOIN DE GRADIENTS! =====
                    with torch.enable_grad():  # ✅ AJOUTER: Activer les gradients
                        logits_support = benchmark.model(train_x)
                        loss_support = benchmark.loss_function(logits_support, train_y)
                        
                        params_adapted = gradient_update_parameters(
                            benchmark.model, 
                            loss_support,
                            step_size=config['step_size'], 
                            params=None,
                            first_order=True
                        )
                    
                    # ===== OUTER LOOP: Evaluation - SANS GRADIENTS =====
                    logits_query = benchmark.model(test_x, params=params_adapted)
                    loss_query = benchmark.loss_function(logits_query, test_y)
                    
                    # Calculer l'accuracy
                    preds = torch.argmax(logits_query, dim=1)
                    accuracy = (preds == test_y).float().mean().item()
                    
                    test_loss += loss_query.item()
                    test_accuracy += accuracy
                    test_count += 1
                    
                    if (test_count) % max(1, config['num_batches'] // 10) == 0:
                        print(f"   Batch {test_count}/{config['num_batches']}: Loss={loss_query.item():.4f}, Acc={accuracy:.4f}")
                
                except Exception as e:
                    print(f"   ❌ ERROR in test batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # ===== RÉSULTATS FINAUX =====
        avg_test_loss = test_loss / test_count if test_count > 0 else 0
        avg_test_accuracy = test_accuracy / test_count if test_count > 0 else 0
        
        results = {
            'test_loss': avg_test_loss,
            'test_accuracy': avg_test_accuracy,
            'test_count': test_count
        }
        
        print(f"\n{'='*80}")
        print(f"🧪 TEST RESULTS")
        print(f"{'='*80}")
        print(f"✅ Test Loss: {avg_test_loss:.4f}")
        print(f"✅ Test Accuracy: {avg_test_accuracy:.4f} ({avg_test_accuracy*100:.2f}%)")
        print(f"✅ Number of test batches: {test_count}")
        print(f"{'='*80}\n")
        
    else:
        # ===== POUR LES STANDARD DATASETS (omniglot, miniimagenet, sinusoid) =====
        print("📊 STANDARD DATASET - Using BatchMetaDataLoader\n")
        
        from torchmeta.utils.data import BatchMetaDataLoader
        
        meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
        metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                                first_order=config['first_order'],
                                                num_adaptation_steps=config['num_steps'],
                                                step_size=config['step_size'],
                                                loss_function=benchmark.loss_function,
                                                device=device)

        results = metalearner.evaluate(meta_test_dataloader,
                                       max_batches=config['num_batches'],
                                       verbose=args.verbose,
                                       desc='Test')
        
        print(f"\n{'='*80}")
        print(f"✅ TEST RESULTS")
        print(f"{'='*80}")
        print(json.dumps(results, indent=2))
        print(f"{'='*80}\n")

    # ===== SAUVEGARDER LES RÉSULTATS =====
    dirname = os.path.dirname(config['model_path'])
    results_path = os.path.join(dirname, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML Test')
    parser.add_argument('config', type=str,
        help='Path to the configuration file returned by `train.py`.')
    parser.add_argument('--folder', type=str, default=None,
        help='Path to the folder the data is downloaded to.')

    # Optimization
    parser.add_argument('--num-steps', type=int, default=-1,
        help='Number of fast adaptation steps.')
    parser.add_argument('--num-batches', type=int, default=-1,
        help='Number of test batches.')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()
    main(args)