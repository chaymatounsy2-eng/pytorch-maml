"""
================================================================================
🧪 HYPERPARAMETER TUNING - ONE-FACTOR-AT-A-TIME (OFAT) APPROACH
================================================================================
✅ AVEC VÉRIFICATION GPU
"""

import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import shutil
import torch  # ✅ NOUVEAU

# ✅ FONCTION POUR VÉRIFIER GPU
def check_gpu_availability():
    """Vérifier si GPU est disponible et configuré"""
    
    print("\n" + "="*80)
    print("🔍 VÉRIFICATION GPU")
    print("="*80)
    
    # Vérifier PyTorch version
    print(f"\n📦 PyTorch version: {torch.__version__}")
    
    # Vérifier CUDA
    cuda_available = torch.cuda.is_available()
    print(f"🔌 CUDA disponible: {'✅ OUI' if cuda_available else '❌ NON'}")
    
    if cuda_available:
        # Nombre de GPUs
        num_gpus = torch.cuda.device_count()
        print(f"📊 Nombre de GPUs: {num_gpus}")
        
        # Détails de chaque GPU
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # GPU actuellement sélectionné
        current_gpu = torch.cuda.current_device()
        print(f"🎯 GPU actuel: GPU {current_gpu}")
        
        # Device
        device = torch.device("cuda")
        print(f"📍 Device: {device}")
        
    else:
        print(f"⚠️  GPU NON disponible, utilisera CPU (TRÈS LENT!)")
        device = torch.device("cpu")
        print(f"📍 Device: {device}")
    
    print("="*80)
    
    return cuda_available, device


# ✅ FONCTION POUR FORCER GPU DANS TRAIN.PY
def add_gpu_flag_to_command(cmd, use_cuda=True):
    """Ajouter flag --use-cuda à la commande training"""
    
    if use_cuda:
        if '--use-cuda' not in cmd:
            cmd += ' --use-cuda'
    
    return cmd


class HyperparameterTuner:
    def __init__(self, base_dir='./pytorch-maml-v2', use_cuda=True):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'hyperparameter_tuning_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # ✅ NOUVEAU: Vérifier GPU
        self.cuda_available, self.device = check_gpu_availability()
        self.use_cuda = use_cuda and self.cuda_available
        
        print(f"\n🚀 Tuner configuré pour utiliser: {'GPU' if self.use_cuda else 'CPU'}")
        
        # Param��tres par défaut
        self.defaults = {
            'lr': 0.002,
            'steps': 5,
            'batch': 4,
            'hidden': 32,
            'epochs': 20,
            'num_ways': 2,
            'num_shots': 5,
            'num_shots_test': 10,
            'step_size': 0.01,
            'num_batches': 100
        }
        
        self.all_results = {
            'timestamp': datetime.now().isoformat(),
            'strategy': 'One-Factor-At-A-Time (OFAT)',
            'gpu_info': {
                'cuda_available': self.cuda_available,
                'device': str(self.device),
                'use_cuda': self.use_cuda
            },
            'stages': {}
        }
        
        print(f"✅ Tuner initialisé")
        print(f"   Base dir: {self.base_dir}")
        print(f"   Results dir: {self.results_dir}")
    
    def train_and_test(self, config, stage_name, test_id):
        """Entraîner et tester avec configuration donnée"""
        
        print(f"\n{'='*80}")
        print(f"🧪 {stage_name} - Test {test_id}")
        print(f"{'='*80}")
        print(f"Config: lr={config['lr']}, steps={config['steps']}, batch={config['batch']}, hidden={config['hidden']}")
        print(f"Device: {self.device}")
        
        output_folder = os.path.join(self.results_dir, f'{stage_name}_test{test_id}')
        os.makedirs(output_folder, exist_ok=True)
        
        # Commande training
        cmd = f"""python train.py ./data_aug_thermal_opt_v2 \
          --dataset thermal \
          --num-ways {config['num_ways']} \
          --num-shots {config['num_shots']} \
          --num-shots-test {config['num_shots_test']} \
          --batch-size {config['batch']} \
          --num-steps {config['steps']} \
          --step-size {config['step_size']} \
          --meta-lr {config['lr']} \
          --num-epochs {config['epochs']} \
          --num-batches {config['num_batches']} \
          --hidden-size {config['hidden']} \
          --output-folder {output_folder} \
          --verbose"""
        
        # ✅ NOUVEAU: Ajouter flag GPU si disponible
        if self.use_cuda:
            cmd += ' --use-cuda'
            print(f"🎯 Flag GPU activé: --use-cuda")
        else:
            print(f"⚠️  GPU non disponible, CPU utilisé (LENT!)")
        
        print(f"\n▶️  Lancement training ({config['epochs']} epochs)...")
        print(f"Output folder: {output_folder}\n")
        
        # Exécuter training
        original_dir = os.getcwd()
        os.chdir(self.base_dir)
        
        try:
            start_time = datetime.now()
            train_result = os.system(cmd)
            elapsed_time = (datetime.now() - start_time).total_seconds() / 60
            
            print(f"\n⏱️  Temps d'entraînement: {elapsed_time:.1f} minutes")
            
            if train_result != 0:
                print("❌ Training échoué!")
                return None
            
            # Commande testing
            config_path = os.path.join(output_folder, 'config.json')
            test_cmd = f"python test.py {config_path}"
            
            # ✅ NOUVEAU: Ajouter flag GPU si disponible
            if self.use_cuda:
                test_cmd += ' --use-cuda'
            
            test_cmd += ' --verbose'
            
            print(f"\n▶️  Lancement testing...")
            test_result = os.system(test_cmd)
            
            if test_result != 0:
                print("❌ Testing échoué!")
                return None
            
            # Charger résultats
            try:
                results_path = os.path.join(output_folder, 'test_results.json')
                with open(results_path, 'r') as f:
                    test_results = json.load(f)
                
                accuracy = test_results.get('accuracy', 0)
                loss = test_results.get('loss', 0)
                
                print(f"\n✅ Test Loss: {loss:.4f}")
                print(f"✅ Test Accuracy: {accuracy:.2%}")
                
                return {
                    'accuracy': accuracy,
                    'loss': loss,
                    'config': config,
                    'output_folder': output_folder,
                    'elapsed_time': elapsed_time
                }
            except Exception as e:
                print(f"❌ Erreur lecture résultats: {e}")
                return None
        
        finally:
            os.chdir(original_dir)
    
    def run_stage1_learning_rate(self):
        """ÉTAPE 1: Chercher meilleur learning_rate"""
        
        print("\n" + "="*80)
        print("🔍 ÉTAPE 1: RECHERCHE DU MEILLEUR LEARNING_RATE")
        print("="*80)
        print(f"Paramètres fixes: steps={self.defaults['steps']}, batch={self.defaults['batch']}, hidden={self.defaults['hidden']}")
        print(f"Device: {self.device}")
        
        lrs_to_test = [0.0005, 0.001, 0.002, 0.005, 0.01]
        results = []
        
        for i, lr in enumerate(lrs_to_test, 1):
            config = self.defaults.copy()
            config['lr'] = lr
            
            result = self.train_and_test(config, 'STAGE1_LR', i)
            
            if result:
                results.append(result)
        
        if not results:
            print("❌ Aucun résultat pour étape 1!")
            return None, []
        
        # Trier par accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Sauvegarder résultats
        stage1_results = {
            'stage': 'ÉTAPE 1: Learning Rate',
            'default_params': {
                'steps': self.defaults['steps'],
                'batch': self.defaults['batch'],
                'hidden': self.defaults['hidden']
            },
            'tested_values': lrs_to_test,
            'results': [
                {
                    'lr': r['config']['lr'],
                    'accuracy': r['accuracy'],
                    'loss': r['loss'],
                    'elapsed_time_minutes': r.get('elapsed_time', 0)
                }
                for r in results
            ],
            'best': {
                'lr': results[0]['config']['lr'],
                'accuracy': results[0]['accuracy'],
                'loss': results[0]['loss']
            }
        }
        
        self.all_results['stages']['stage1'] = stage1_results
        
        print(f"\n{'='*80}")
        print("🏆 RÉSULTATS ÉTAPE 1 - TOP 3 LEARNING RATES:")
        print(f"{'='*80}")
        for i, result in enumerate(results[:3], 1):
            time_str = f" ({result.get('elapsed_time', 0):.1f} min)" if 'elapsed_time' in result else ""
            print(f"{i}. lr={result['config']['lr']:.6f}, Accuracy={result['accuracy']:.2%}, Loss={result['loss']:.4f}{time_str}")
        
        best_lr = results[0]['config']['lr']
        print(f"\n✅ MEILLEUR LR SÉLECTIONNÉ: {best_lr}")
        
        return best_lr, results
    
    def run_stage2_num_steps(self, best_lr):
        """ÉTAPE 2: Chercher meilleur num_steps"""
        
        print("\n" + "="*80)
        print("🔍 ÉTAPE 2: RECHERCHE DU MEILLEUR NUM_STEPS")
        print(f"   (avec learning_rate={best_lr})")
        print("="*80)
        print(f"Paramètres fixes: lr={best_lr}, batch={self.defaults['batch']}, hidden={self.defaults['hidden']}")
        print(f"Device: {self.device}")
        
        steps_to_test = [1, 2, 3, 5, 7, 10]
        results = []
        
        for i, steps in enumerate(steps_to_test, 1):
            config = self.defaults.copy()
            config['lr'] = best_lr
            config['steps'] = steps
            
            result = self.train_and_test(config, 'STAGE2_STEPS', i)
            
            if result:
                results.append(result)
        
        if not results:
            print("❌ Aucun résultat pour étape 2!")
            return None, []
        
        # Trier par accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Sauvegarder résultats
        stage2_results = {
            'stage': 'ÉTAPE 2: Num Steps',
            'fixed_params': {
                'lr': best_lr,
                'batch': self.defaults['batch'],
                'hidden': self.defaults['hidden']
            },
            'tested_values': steps_to_test,
            'results': [
                {
                    'steps': r['config']['steps'],
                    'accuracy': r['accuracy'],
                    'loss': r['loss'],
                    'elapsed_time_minutes': r.get('elapsed_time', 0)
                }
                for r in results
            ],
            'best': {
                'steps': results[0]['config']['steps'],
                'accuracy': results[0]['accuracy'],
                'loss': results[0]['loss']
            }
        }
        
        self.all_results['stages']['stage2'] = stage2_results
        
        print(f"\n{'='*80}")
        print("🏆 RÉSULTATS ÉTAPE 2 - TOP 3 NUM_STEPS:")
        print(f"{'='*80}")
        for i, result in enumerate(results[:3], 1):
            time_str = f" ({result.get('elapsed_time', 0):.1f} min)" if 'elapsed_time' in result else ""
            print(f"{i}. steps={result['config']['steps']}, Accuracy={result['accuracy']:.2%}, Loss={result['loss']:.4f}{time_str}")
        
        best_steps = results[0]['config']['steps']
        print(f"\n✅ MEILLEUR STEPS SÉLECTIONNÉ: {best_steps}")
        
        return best_steps, results
    
    def run_stage3_batch_size(self, best_lr, best_steps):
        """ÉTAPE 3: Chercher meilleur batch_size"""
        
        print("\n" + "="*80)
        print("🔍 ÉTAPE 3: RECHERCHE DU MEILLEUR BATCH_SIZE")
        print(f"   (avec learning_rate={best_lr}, num_steps={best_steps})")
        print("="*80)
        print(f"Paramètres fixes: lr={best_lr}, steps={best_steps}, hidden={self.defaults['hidden']}")
        print(f"Device: {self.device}")
        
        batches_to_test = [2, 4, 8, 16]
        results = []
        
        for i, batch in enumerate(batches_to_test, 1):
            config = self.defaults.copy()
            config['lr'] = best_lr
            config['steps'] = best_steps
            config['batch'] = batch
            
            result = self.train_and_test(config, 'STAGE3_BATCH', i)
            
            if result:
                results.append(result)
        
        if not results:
            print("❌ Aucun résultat pour étape 3!")
            return None, []
        
        # Trier par accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Sauvegarder résultats
        stage3_results = {
            'stage': 'ÉTAPE 3: Batch Size',
            'fixed_params': {
                'lr': best_lr,
                'steps': best_steps,
                'hidden': self.defaults['hidden']
            },
            'tested_values': batches_to_test,
            'results': [
                {
                    'batch': r['config']['batch'],
                    'accuracy': r['accuracy'],
                    'loss': r['loss'],
                    'elapsed_time_minutes': r.get('elapsed_time', 0)
                }
                for r in results
            ],
            'best': {
                'batch': results[0]['config']['batch'],
                'accuracy': results[0]['accuracy'],
                'loss': results[0]['loss']
            }
        }
        
        self.all_results['stages']['stage3'] = stage3_results
        
        print(f"\n{'='*80}")
        print("🏆 RÉSULTATS ÉTAPE 3 - TOP 3 BATCH_SIZES:")
        print(f"{'='*80}")
        for i, result in enumerate(results[:3], 1):
            time_str = f" ({result.get('elapsed_time', 0):.1f} min)" if 'elapsed_time' in result else ""
            print(f"{i}. batch={result['config']['batch']}, Accuracy={result['accuracy']:.2%}, Loss={result['loss']:.4f}{time_str}")
        
        best_batch = results[0]['config']['batch']
        print(f"\n✅ MEILLEUR BATCH_SIZE SÉLECTIONNÉ: {best_batch}")
        
        return best_batch, results
    
    def run_stage4_hidden_size(self, best_lr, best_steps, best_batch):
        """ÉTAPE 4: Chercher meilleur hidden_size"""
        
        print("\n" + "="*80)
        print("🔍 ÉTAPE 4: RECHERCHE DU MEILLEUR HIDDEN_SIZE")
        print(f"   (avec lr={best_lr}, steps={best_steps}, batch={best_batch})")
        print("="*80)
        print(f"Paramètres fixes: lr={best_lr}, steps={best_steps}, batch={best_batch}")
        print(f"Device: {self.device}")
        
        hiddens_to_test = [16, 32, 64, 128]
        results = []
        
        for i, hidden in enumerate(hiddens_to_test, 1):
            config = self.defaults.copy()
            config['lr'] = best_lr
            config['steps'] = best_steps
            config['batch'] = best_batch
            config['hidden'] = hidden
            
            result = self.train_and_test(config, 'STAGE4_HIDDEN', i)
            
            if result:
                results.append(result)
        
        if not results:
            print("❌ Aucun résultat pour étape 4!")
            return None, []
        
        # Trier par accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Sauvegarder résultats
        stage4_results = {
            'stage': 'ÉTAPE 4: Hidden Size',
            'fixed_params': {
                'lr': best_lr,
                'steps': best_steps,
                'batch': best_batch
            },
            'tested_values': hiddens_to_test,
            'results': [
                {
                    'hidden': r['config']['hidden'],
                    'accuracy': r['accuracy'],
                    'loss': r['loss'],
                    'elapsed_time_minutes': r.get('elapsed_time', 0)
                }
                for r in results
            ],
            'best': {
                'hidden': results[0]['config']['hidden'],
                'accuracy': results[0]['accuracy'],
                'loss': results[0]['loss']
            }
        }
        
        self.all_results['stages']['stage4'] = stage4_results
        
        print(f"\n{'='*80}")
        print("🏆 RÉSULTATS ÉTAPE 4 - TOP 3 HIDDEN_SIZES:")
        print(f"{'='*80}")
        for i, result in enumerate(results[:3], 1):
            time_str = f" ({result.get('elapsed_time', 0):.1f} min)" if 'elapsed_time' in result else ""
            print(f"{i}. hidden={result['config']['hidden']}, Accuracy={result['accuracy']:.2%}, Loss={result['loss']:.4f}{time_str}")
        
        best_hidden = results[0]['config']['hidden']
        print(f"\n✅ MEILLEUR HIDDEN_SIZE SÉLECTIONNÉ: {best_hidden}")
        
        return best_hidden, results
    
    def generate_comparison_plots(self):
        """Générer graphiques de comparaison"""
        
        print("\n" + "="*80)
        print("📊 GÉNÉRATION DES GRAPHIQUES")
        print("="*80)
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Hyperparameter Tuning Results - OFAT Approach (Device: {self.device})', fontsize=16, fontweight='bold')
            
            # STAGE 1: Learning Rate
            if 'stage1' in self.all_results['stages']:
                stage1 = self.all_results['stages']['stage1']
                lrs = [r['lr'] for r in stage1['results']]
                accs = [r['accuracy'] for r in stage1['results']]
                
                axes[0, 0].plot(range(len(lrs)), accs, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
                axes[0, 0].set_xticks(range(len(lrs)))
                axes[0, 0].set_xticklabels([f'{lr:.4f}' for lr in lrs], rotation=45)
                axes[0, 0].set_ylabel('Accuracy', fontsize=12)
                axes[0, 0].set_xlabel('Learning Rate', fontsize=12)
                axes[0, 0].set_title('ÉTAPE 1: Learning Rate Tuning', fontsize=12, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_ylim([0, 1])
            
            # STAGE 2: Num Steps
            if 'stage2' in self.all_results['stages']:
                stage2 = self.all_results['stages']['stage2']
                steps = [r['steps'] for r in stage2['results']]
                accs = [r['accuracy'] for r in stage2['results']]
                
                axes[0, 1].plot(range(len(steps)), accs, 's-', linewidth=2, markersize=8, color='#4ECDC4')
                axes[0, 1].set_xticks(range(len(steps)))
                axes[0, 1].set_xticklabels(steps)
                axes[0, 1].set_ylabel('Accuracy', fontsize=12)
                axes[0, 1].set_xlabel('Num Steps', fontsize=12)
                axes[0, 1].set_title('ÉTAPE 2: Num Steps Tuning', fontsize=12, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_ylim([0, 1])
            
            # STAGE 3: Batch Size
            if 'stage3' in self.all_results['stages']:
                stage3 = self.all_results['stages']['stage3']
                batches = [r['batch'] for r in stage3['results']]
                accs = [r['accuracy'] for r in stage3['results']]
                
                axes[1, 0].plot(range(len(batches)), accs, '^-', linewidth=2, markersize=8, color='#95E1D3')
                axes[1, 0].set_xticks(range(len(batches)))
                axes[1, 0].set_xticklabels(batches)
                axes[1, 0].set_ylabel('Accuracy', fontsize=12)
                axes[1, 0].set_xlabel('Batch Size', fontsize=12)
                axes[1, 0].set_title('ÉTAPE 3: Batch Size Tuning', fontsize=12, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim([0, 1])
            
            # STAGE 4: Hidden Size
            if 'stage4' in self.all_results['stages']:
                stage4 = self.all_results['stages']['stage4']
                hiddens = [r['hidden'] for r in stage4['results']]
                accs = [r['accuracy'] for r in stage4['results']]
                
                axes[1, 1].plot(range(len(hiddens)), accs, 'd-', linewidth=2, markersize=8, color='#F38181')
                axes[1, 1].set_xticks(range(len(hiddens)))
                axes[1, 1].set_xticklabels(hiddens)
                axes[1, 1].set_ylabel('Accuracy', fontsize=12)
                axes[1, 1].set_xlabel('Hidden Size', fontsize=12)
                axes[1, 1].set_title('ÉTAPE 4: Hidden Size Tuning', fontsize=12, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim([0, 1])
            
            plt.tight_layout()
            
            plot_path = os.path.join(self.results_dir, 'hyperparameter_tuning_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Graphique sauvegardé: {plot_path}")
            
            plt.close()
        
        except Exception as e:
            print(f"⚠️  Erreur génération graphiques: {e}")
    
    def print_final_summary(self):
        """Afficher résumé final"""
        
        print("\n" + "="*80)
        print("🏆 RÉSUMÉ FINAL - HYPERPARAMETERS OPTIMISÉS")
        print("="*80)
        
        if 'stage1' not in self.all_results['stages']:
            print("❌ Aucune donnée disponible!")
            return
        
        best_lr = self.all_results['stages']['stage1']['best']['lr']
        best_steps = self.all_results['stages']['stage2']['best']['steps'] if 'stage2' in self.all_results['stages'] else self.defaults['steps']
        best_batch = self.all_results['stages']['stage3']['best']['batch'] if 'stage3' in self.all_results['stages'] else self.defaults['batch']
        best_hidden = self.all_results['stages']['stage4']['best']['hidden'] if 'stage4' in self.all_results['stages'] else self.defaults['hidden']
        
        print(f"""
🎯 MEILLEURS HYPERPARAMETERS TROUVÉS:
   • Learning Rate: {best_lr}
   • Num Steps: {best_steps}
   • Batch Size: {best_batch}
   • Hidden Size: {best_hidden}

📊 ACCURACIES À CHAQUE ÉTAPE:
   • ÉTAPE 1 (LR): {self.all_results['stages']['stage1']['best']['accuracy']:.2%}
   • ÉTAPE 2 (STEPS): {self.all_results['stages']['stage2']['best']['accuracy']:.2%}
   • ÉTAPE 3 (BATCH): {self.all_results['stages']['stage3']['best']['accuracy']:.2%}
   • ÉTAPE 4 (HIDDEN): {self.all_results['stages']['stage4']['best']['accuracy']:.2%}

💻 DEVICE UTILISÉ:
   • Device: {self.device}
   • CUDA Available: {self.cuda_available}
   • Temps estimé avec GPU: 60-90 minutes
   • Temps estimé avec CPU: 240-360 minutes

💡 PROCHAINE ÉTAPE:
   Entraîner avec ces paramètres pour 150 epochs (validation complète)
   
   Command:
   python train.py ./data_aug_thermal_opt_v2 \\
     --meta-lr {best_lr} \\
     --num-steps {best_steps} \\
     --batch-size {best_batch} \\
     --hidden-size {best_hidden} \\
     --num-epochs 150 \\
     {'--use-cuda' if self.use_cuda else ''}
        """)
    
    def run_complete_tuning(self):
        """Lancer le tuning complet"""
        
        print("\n" + "="*80)
        print("🚀 DÉMARRAGE HYPERPARAMETER TUNING - STRATÉGIE OFAT")
        print("="*80)
        print(f"Timestamp: {self.all_results['timestamp']}")
        print(f"Résultats: {self.results_dir}")
        print(f"Device: {self.device}")
        print(f"CUDA Available: {self.cuda_available}")
        
        try:
            # ÉTAPE 1
            best_lr, results_lr = self.run_stage1_learning_rate()
            if best_lr is None:
                print("❌ Étape 1 échouée!")
                return None
            
            # ÉTAPE 2
            best_steps, results_steps = self.run_stage2_num_steps(best_lr)
            if best_steps is None:
                print("❌ Étape 2 échouée!")
                return None
            
            # ÉTAPE 3
            best_batch, results_batch = self.run_stage3_batch_size(best_lr, best_steps)
            if best_batch is None:
                print("❌ Étape 3 échouée!")
                return None
            
            # ÉTAPE 4
            best_hidden, results_hidden = self.run_stage4_hidden_size(best_lr, best_steps, best_batch)
            if best_hidden is None:
                print("❌ Étape 4 échouée!")
                return None
            
            # Ajouter meilleure config finale
            self.all_results['best_config'] = {
                'lr': best_lr,
                'steps': best_steps,
                'batch': best_batch,
                'hidden': best_hidden,
                'expected_accuracy': self.all_results['stages']['stage4']['best']['accuracy']
            }
            
            # Sauvegarder résultats JSON
            results_path = os.path.join(self.results_dir, 'complete_tuning_results.json')
            with open(results_path, 'w') as f:
                json.dump(self.all_results, f, indent=2)
            print(f"\n✅ Résultats sauvegardés: {results_path}")
            
            # Générer graphiques
            self.generate_comparison_plots()
            
            # Afficher résumé
            self.print_final_summary()
            
            print("\n" + "="*80)
            print("✅ TUNING COMPLÉTÉ!")
            print("="*80)
            
            return self.all_results
        
        except Exception as e:
            print(f"\n❌ ERREUR PENDANT TUNING: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # ✅ CORRIGÉ: Utiliser '.' au lieu de './pytorch-maml-v2'
    base_path = '.'  # Vous êtes déjà dans pytorch-maml-v2
    
    if not os.path.exists(base_path):
        print(f"❌ Le dossier {base_path} n'existe pas!")
        print("Chemins disponibles:")
        print(os.listdir('.'))
        exit(1)
    
    # ✅ NOUVEAU: Paramètre pour forcer/désactiver GPU
    use_cuda = True  # Mettre False pour forcer CPU
    
    # Créer tuner
    tuner = HyperparameterTuner(base_dir=base_path, use_cuda=use_cuda)
    
    # Lancer tuning complet
    results = tuner.run_complete_tuning()
    
    if results:
        print("\n✅ Tuning réussi!")
    else:
        print("\n❌ Tuning échoué!")