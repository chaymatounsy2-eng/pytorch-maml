import torch
import math
import os
import time
import json
import logging

from torchmeta.utils.data import BatchMetaDataLoader
from torch.utils.data import DataLoader
from torchmeta.utils import gradient_update_parameters

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning

# Désactiver le debug logging de PIL
logging.getLogger('PIL.TiffImagePlugin').setLevel(logging.WARNING)

# ============================================================================
# CLASSE EarlyStopping
# ============================================================================
class EarlyStopping:
    """
    Early stopping pour arrêter l'entraînement si la validation loss ne s'améliore pas
    """
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        """
        Args:
            patience: Nombre d'epochs avant d'arrêter si pas d'amélioration
            min_delta: Amélioration minimale considérée comme valide
            verbose: Afficher les messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Vérifier si on doit arrêter
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            # S'améliore suffisamment
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"✅ Amélioration détectée! Nouveau meilleur: {val_loss:.4f}")
        else:
            # Pas d'amélioration
            self.counter += 1
            if self.verbose:
                print(f"⚠️  Pas d'amélioration ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 EARLY STOPPING! Pas d'amélioration pendant {self.patience} epochs")


def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"🚀 MAML TRAINING STARTED")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            logging.debug('Creating folder `{0}`'.format(args.output_folder))

        folder = os.path.join(args.output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)
        logging.debug('Creating folder `{0}`'.format(folder))

        args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))

    print("📦 Loading benchmark...")
    benchmark = get_benchmark_by_name(args.dataset,
                                      args.folder,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      hidden_size=args.hidden_size)
    print("✅ Benchmark loaded!\n")

    # =====================================================================
    # THERMAL DATASET: Custom training loop
    # =====================================================================
    if args.dataset == 'thermal':
        print("🔥 THERMAL DATASET DETECTED - Using custom training loop\n")
        
        # Create simple dataloaders
        print("📦 Creating train dataloader...")
        meta_train_dataloader = DataLoader(benchmark.meta_train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=0)
        
        print("📦 Creating val dataloader...")
        meta_val_dataloader = DataLoader(benchmark.meta_val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=0)
        
        print("✅ DataLoaders created!\n")
        
        # Create optimizer and metalearner
        print("⚙️  Creating optimizer and metalearner...")
        meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)
        metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                                optimizer=meta_optimizer,
                                                first_order=args.first_order,
                                                num_adaptation_steps=args.num_steps,
                                                step_size=args.step_size,
                                                loss_function=benchmark.loss_function,
                                                device=device)
        print("✅ Metalearner created!\n")
        
        # Create EarlyStopping
        early_stopping = EarlyStopping(patience=15, min_delta=0.001, verbose=True)
        
        # CUSTOM TRAINING LOOP FOR THERMAL
        best_value = None
        
        for epoch in range(args.num_epochs):
            print(f"\n{'='*80}")
            print(f"📅 EPOCH {epoch + 1}/{args.num_epochs}")
            print(f"{'='*80}")
            
            # ============= TRAIN =============
            print("🔵 TRAINING...")
            num_batches = 0
            total_loss = 0.0
            total_accuracy = 0.0  # ✅ AJOUTER
            
            for batch_idx, batch in enumerate(meta_train_dataloader):
                if num_batches >= args.num_batches:
                    print(f"   ✅ Max batches reached ({args.num_batches})")
                    break
                
                try:
                    # Each batch contains one complete MAML task
                    train_x, train_y = batch['train']
                    test_x, test_y = batch['test']
                    
                    # Reshape to (N, 3, 84, 84)
                    train_x = train_x.view(-1, 3, 84, 84).to(device)
                    train_y = train_y.view(-1).to(device)
                    test_x = test_x.view(-1, 3, 84, 84).to(device)
                    test_y = test_y.view(-1).to(device)
                    
                    # ===== INNER LOOP: Adaptation =====
                    # Forward on support set
                    logits_support = benchmark.model(train_x)
                    loss_support = benchmark.loss_function(logits_support, train_y)
                    
                    # Adapt with gradient_update_parameters
                    params_adapted = gradient_update_parameters(
                        benchmark.model, 
                        loss_support,
                        step_size=args.step_size, 
                        params=None,
                        first_order=args.first_order
                    )
                    
                    # ===== OUTER LOOP: Evaluation =====
                    # Evaluate on query set with adapted weights
                    with torch.set_grad_enabled(benchmark.model.training):
                        logits_query = benchmark.model(test_x, params=params_adapted)
                        loss_query = benchmark.loss_function(logits_query, test_y)
                    
                    # ✅ AJOUTER: Calculer accuracy
                    preds = torch.argmax(logits_query, dim=1)
                    accuracy = (preds == test_y).float().mean().item()
                    
                    # Meta-update
                    metalearner.optimizer.zero_grad()
                    loss_query.backward()
                    metalearner.optimizer.step()
                    
                    total_loss += loss_query.item()
                    total_accuracy += accuracy  # ✅ AJOUTER
                    num_batches += 1
                    
                    if (num_batches) % max(1, args.num_batches // 10) == 0:
                        print(f"   Batch {num_batches}/{args.num_batches}: Loss={loss_query.item():.4f}, Acc={accuracy:.4f}")
                
                except Exception as e:
                    print(f"   ❌ ERROR in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_train_accuracy = total_accuracy / num_batches if num_batches > 0 else 0  # ✅ AJOUTER
            print(f"✅ Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}")
            
            # ============= VALIDATION =============
            print("🟡 VALIDATING...")
            val_loss = 0.0
            val_accuracy = 0.0  # ✅ AJOUTER
            val_count = 0
            
            # ✅ CORRIGÉ: Enlever torch.no_grad() du wrapper principal
            for batch in meta_val_dataloader:
                if val_count >= args.num_batches:
                    print(f"   ✅ Max validation batches reached ({args.num_batches})")
                    break
                
                try:
                    train_x, train_y = batch['train']
                    test_x, test_y = batch['test']
                    
                    train_x = train_x.view(-1, 3, 84, 84).to(device)
                    train_y = train_y.view(-1).to(device)
                    test_x = test_x.view(-1, 3, 84, 84).to(device)
                    test_y = test_y.view(-1).to(device)
                    
                    # ===== INNER LOOP: Adaptation (AVEC gradients) =====
                    logits_support = benchmark.model(train_x)
                    loss_support = benchmark.loss_function(logits_support, train_y)
                    
                    params_adapted = gradient_update_parameters(
                        benchmark.model, 
                        loss_support,
                        step_size=args.step_size, 
                        params=None,
                        first_order=True
                    )
                    
                    # ===== OUTER LOOP: Evaluation (SANS gradients) =====
                    with torch.no_grad():  # ✅ Seulement ici
                        logits_query = benchmark.model(test_x, params=params_adapted)
                        loss_query = benchmark.loss_function(logits_query, test_y)
                        
                        # ✅ AJOUTER: Calculer accuracy
                        preds = torch.argmax(logits_query, dim=1)
                        accuracy = (preds == test_y).float().mean().item()
                    
                    val_loss += loss_query.item()
                    val_accuracy += accuracy  # ✅ AJOUTER
                    val_count += 1
                
                except Exception as e:
                    print(f"   ❌ ERROR in validation batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            avg_val_loss = val_loss / val_count if val_count > 0 else 0
            avg_val_accuracy = val_accuracy / val_count if val_count > 0 else 0  # ✅ AJOUTER
            print(f"✅ Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}")  # ✅ MODIFIÉ
            
            # Save best model
            if best_value is None or best_value > avg_val_loss:
                best_value = avg_val_loss
                if args.output_folder is not None:
                    with open(args.model_path, 'wb') as f:
                        torch.save(benchmark.model.state_dict(), f)
                    print(f"💾 Model saved! Best val loss: {avg_val_loss:.4f}, Acc: {avg_val_accuracy:.4f}")  # ✅ MODIFIÉ
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0 and args.output_folder is not None:
                checkpoint_path = args.model_path.replace('model.th', f'checkpoint_epoch_{epoch+1}.th')
                with open(checkpoint_path, 'wb') as f:
                    torch.save(benchmark.model.state_dict(), f)
                print(f"📌 Checkpoint saved at epoch {epoch + 1}")
            
            # ============= EARLY STOPPING CHECK =============
            early_stopping(avg_val_loss)
            
            if early_stopping.early_stop:
                print(f"\n{'='*80}")
                print(f"🛑 TRAINING STOPPED EARLY AT EPOCH {epoch + 1}")
                print(f"{'='*80}\n")
                break
        
        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETED!")
        print(f"{'='*80}\n")
        return
    
    # =====================================================================
    # STANDARD DATASETS (omniglot, miniimagenet, sinusoid)
    # =====================================================================
    print("📊 STANDARD DATASET - Using BatchMetaDataLoader\n")
    
    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)
    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            optimizer=meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            step_size=args.step_size,
                                            loss_function=benchmark.loss_function,
                                            device=device)

    best_value = None

    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    for epoch in range(args.num_epochs):
        metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))

        # Save best model
        if 'accuracies_after' in results:
            if (best_value is None) or (best_value < results['accuracies_after']):
                best_value = results['accuracies_after']
                save_model = True
        elif (best_value is None) or (best_value > results['mean_outer_loss']):
            best_value = results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)

    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet', 'thermal'], 
        default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)