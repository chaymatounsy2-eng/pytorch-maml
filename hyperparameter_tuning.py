"""
================================================================================
🧪 HYPERPARAMETER TUNING - ONE-FACTOR-AT-A-TIME (OFAT) APPROACH
================================================================================

✅ VERSION CORRIGÉE - SPLIT TRAIN/VAL/TEST CORRECT

Stratégie:
  TUNING (utilise meta-val):
    - ÉTAPE 1: Varier learning_rate
    - ÉTAPE 2: Varier num_steps
    - ÉTAPE 3: Varier batch_size
    - ÉTAPE 4: Varier hidden_size

  TEST FINAL (utilise meta-test - UNE SEULE FOIS):
    - Entraîner avec meilleurs paramètres
    - Évaluer sur meta-test (jamais vu avant!)
"""

import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import shutil
import torch

# ✅ FONCTION POUR VÉRIFIER GPU
def check_gpu_availability():
    """Vérifier si GPU est disponible et configuré"""
    
    print("\n" + "="*80)
    print("🔍 VÉRIFICATION GPU")
    print("="*80)
    
    print(f"\n📦 PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"🔌 CUDA disponible: {'✅ OUI' if cuda_available else '❌ NON'}")
    
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"📊 Nombre de GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        current_gpu = torch.cuda.current_device()
        print(f"🎯 GPU actuel: GPU {current_gpu}")
        
        device = torch.device("cuda")
        print(f"📍 Device: {device}")
        
    else:
        print(f"⚠️  GPU NON disponible, utilisera CPU (TRÈS LENT!)")
        device = torch.device("cpu")
        print(f"📍 Device: {device}")
    
    print("="*80)
    
    return cuda_available, device


class HyperparameterTuner:
    def __init__(self, base_dir='.', use_cuda=True):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'hyperparameter_tuning_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # ✅ Vérifier GPU
        self.cuda_available, self.device = check_gpu_availability()
        self.use_cuda = use_cuda and self.cuda_available
        
        print(f"\n🚀 Tuner configuré pour utiliser: {'GPU' if self.use_cuda else 'CPU'}")
        
        # Paramètres par défaut
        self.defaults = {
            'lr': 0.002,
            'steps': 5,
            'batch': 4,
            'hidden': 64,  # ✅ CHANGÉ À 64
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
            'description': 'Tuning avec meta-val, test final avec meta-test',
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
    
    def train_and_validate(self, config, stage_name, test_id):
        """
        ✅ ENTRAÎNER ET VALIDER AVEC META-VAL (PAS META-TEST!)
        
        1. Train sur meta-train
        2. Évaluer sur meta-val
        3. Retourner metrics de validation
        """
        
        print(f"\n{'='*80}")
        print(f"🧪 {stage_name} - Test {test_id}")
        print(f"{'='*80}")
        print(f"Config: lr={config['lr']}, steps={config['steps']}, batch={config['batch']}, hidden={config['hidden']}")
        print(f"Device: {self.device}")
        print(f"📊 Évaluation sur: META-VAL (pour le tuning)")
        
        output_folder = os.path.join(self.results_dir, f'{stage_name}_test{test_id}')
        os.makedirs(output_folder, exist_ok=True)
        
        # ✅ Commande training sur meta-train
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
        
        if self.use_cuda:
            cmd += ' --use-cuda'
            print(f"🎯 Flag GPU activé: --use-cuda")
        
        print(f"\n▶️  Lancement TRAINING sur meta-train ({config['epochs']} epochs)...")
        print(f"Output folder: {output_folder}\n")
        
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
            
            # ✅ VALIDATION sur meta-val (PAS meta-test!)
            config_path = os.path.join(output_folder, 'config.json')
            val_cmd = f"python test.py {config_path}"
            
            if self.use_cuda:
                val_cmd += ' --use-cuda'
            
            val_cmd += ' --verbose'
            
            print(f"\n▶️  Lancement VALIDATION sur meta-val...")
            val_result = os.system(val_cmd)
            
            if val_result != 0:
                print("❌ Validation échouée!")
                return None
            
            # ✅ Charger résultats de validation
            try:
                val_results_path = os.path.join(output_folder, 'test_results.json')
                with open(val_results_path, 'r') as f:
                    val_results = json.load(f)
                
                accuracy = val_results.get('accuracy', 0)
                loss = val_results.get('loss', 0)
                
                print(f"\n✅ Val Loss: {loss:.4f}")
                print(f"✅ Val Accuracy: {accuracy:.2%}")
                
                return {
                    'accuracy': accuracy,
                    'loss': loss,
                    'config': config,
                    'output_folder': output_folder,
                    'elapsed_time': elapsed_time
                }
            except Exception as e:
                print(f"❌ Erreur lecture résultats validation: {e}")
                return None
        
        finally:
            os.chdir(original_dir)
    
    def run_stage1_learning_rate(self):
        """ÉTAPE 1: Chercher meilleur learning_rate (sur meta-val)"""
        
        print("\n" + "="*80)
        print("🔍 ÉTAPE 1: RECHERCHE DU MEILLEUR LEARNING_RATE")
        print("="*80)
        print(f"Paramètres fixes: steps={self.defaults['steps']}, batch={self.defaults['batch']}, hidden={self.defaults['hidden']}")
        print(f"Device: {self.device}")
        print(f"📊 Évaluation sur: META-VAL")
        
        lrs_to_test = [0.0005, 0.001, 0.002, 0.005, 0.01]
        results = []
        
        for i, lr in enumerate(lrs_to_test, 1):
            config = self.defaults.copy()
            config['lr'] = lr
            
            result = self.train_and_validate(config, 'STAGE1_LR', i)
            
            if result:
                results.append(result)
        
        if not results:
            print("❌ Aucun résultat pour étape 1!")
            return None, []
        
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        stage1_results = {
            'stage': 'ÉTAPE 1: Learning Rate',
            'evaluation_set': 'meta-val',
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
            print(f"{i}. lr={result['config']['lr']:.6f}, Val Accuracy={result['accuracy']:.2%}, Loss={result['loss']:.4f}{time_str}")
        
        best_lr = results[0]['config']['lr']
        print(f"\n✅ MEILLEUR LR SÉLECTIONNÉ: {best_lr}")
        
        return best_lr, results
    
    def run_stage2_num_steps(self, best_lr):
        """ÉTAPE 2: Chercher meilleur num_steps (sur meta-val)"""
        
        print("\n" + "="*80)
        print("🔍 ÉTAPE 2: RECHERCHE DU MEILLEUR NUM_STEPS")
        print(f"   (avec learning_rate={best_lr})")
        print("="*80)
        print(f"Paramètres fixes: lr={best_lr}, batch={self.defaults['batch']}, hidden={self.defaults['hidden']}")
        print(f"Device: {self.device}")
        print(f"📊 Évaluation sur: META-VAL")
        
        steps_to_test = [1, 2, 3, 5, 7, 10]
        results = []
        
        for i, steps in enumerate(steps_to_test, 1):
            config = self.defaults.copy()
            config['lr'] = best_lr
            config['steps'] = steps
            
            result = self.train_and_validate(config, 'STAGE2_STEPS', i)
            
            if result:
                results.append(result)
        
        if not results:
            print("❌ Aucun résultat pour étape 2!")
            return None, []
        
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        stage2_results = {
            'stage': 'ÉTAPE 2: Num Steps',
            'evaluation_set': 'meta-val',
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
            print(f"{i}. steps={result['config']['steps']}, Val Accuracy={result['accuracy']:.2%}, Loss={result['loss']:.4f}{time_str}")
        
        best_steps = results[0]['config']['steps']
        print(f"\n✅ MEILLEUR STEPS SÉLECTIONNÉ: {best_steps}")
        
        return best_steps, results
    
    def run_stage3_batch_size(self, best_lr, best_steps):
        """ÉTAPE 3: Chercher meilleur batch_size (sur meta-val)"""
        
        print("\n" + "="*80)
        print("🔍 ÉTAPE 3: RECHERCHE DU MEILLEUR BATCH_SIZE")
        print(f"   (avec learning_rate={best_lr}, num_steps={best_steps})")
        print("="*80)
        print(f"Paramètres fixes: lr={best_lr}, steps={best_steps}, hidden={self.defaults['hidden']}")
        print(f"Device: {self.device}")
        print(f"📊 Évaluation sur: META-VAL")
        
        batches_to_test = [2, 4, 8, 16]
        results = []
        
        for i, batch in enumerate(batches_to_test, 1):
            config = self.defaults.copy()
            config['lr'] = best_lr
            config['steps'] = best_steps
            config['batch'] = batch
            
            result = self.train_and_validate(config, 'STAGE3_BATCH', i)
            
            if result:
                results.append(result)
        
        if not results:
            print("❌ Aucun résultat pour étape 3!")
            return None, []
        
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        stage3_results = {
            'stage': 'ÉTAPE 3: Batch Size',
            'evaluation_set': 'meta-val',
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
            print(f"{i}. batch={result['config']['batch']}, Val Accuracy={result['accuracy']:.2%}, Loss={result['loss']:.4f}{time_str}")
        
        best_batch = results[0]['config']['batch']
        print(f"\n✅ MEILLEUR BATCH_SIZE SÉLECTIONNÉ: {best_batch}")
        
        return best_batch, results
    
    def run_stage4_hidden_size(self, best_lr, best_steps, best_batch):
        """ÉTAPE 4: Chercher meilleur hidden_size (sur meta-val)"""
        
        print("\n" + "="*80)
        print("🔍 ÉTAPE 4: RECHERCHE DU MEILLEUR HIDDEN_SIZE")
        print(f"   (avec lr={best_lr}, steps={best_steps}, batch={best_batch})")
        print("="*80)
        print(f"Paramètres fixes: lr={best_lr}, steps={best_steps}, batch={best_batch}")
        print(f"Device: {self.device}")
        print(f"📊 Évaluation sur: META-VAL")
        
        hiddens_to_test = [16, 32, 64, 128]
        results = []
        
        for i, hidden in enumerate(hiddens_to_test, 1):
            config = self.defaults.copy()
            config['lr'] = best_lr
            config['steps'] = best_steps
            config['batch'] = best_batch
            config['hidden'] = hidden
            
            result = self.train_and_validate(config, 'STAGE4_HIDDEN', i)
            
            if result:
                results.append(result)
        
        if not results:
            print("❌ Aucun résultat pour étape 4!")
            return None, []
        
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        stage4_results = {
            'stage': 'ÉTAPE 4: Hidden Size',
            'evaluation_set': 'meta-val',
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
            print(f"{i}. hidden={result['config']['hidden']}, Val Accuracy={result['accuracy']:.2%}, Loss={result['loss']:.4f}{time_str}")
        
        best_hidden = results[0]['config']['hidden']
        print(f"\n✅ MEILLEUR HIDDEN_SIZE SÉLECTIONNÉ: {best_hidden}")
        
        return best_hidden, results
    
    def run_final_test(self, best_config):
        """
        ✅ TEST FINAL - ÉVALUER SUR META-TEST UNE SEULE FOIS!
        
        Avec la meilleure configuration trouvée pendant le tuning
        """
        
        print("\n" + "="*80)
        print("🏆 TEST FINAL - ÉVALUATION SUR META-TEST")
        print("="*80)
        print(f"\n📊 Évaluation sur: META-TEST (JAMAIS VU AVANT!)")
        print(f"\nConfiguration sélectionnée (des étapes 1-4):")
        print(f"  • Learning Rate: {best_config['lr']}")
        print(f"  • Num Steps: {best_config['steps']}")
        print(f"  • Batch Size: {best_config['batch']}")
        print(f"  • Hidden Size: {best_config['hidden']}")
        
        final_output_folder = os.path.join(self.results_dir, 'FINAL_TEST')
        os.makedirs(final_output_folder, exist_ok=True)
        
        # ✅ Entraîner COMPLÈTEMENT avec meilleure config (150 epochs)
        cmd = f"""python train.py ./data_aug_thermal_opt_v2 \
          --dataset thermal \
          --num-ways {best_config['num_ways']} \
          --num-shots {best_config['num_shots']} \
          --num-shots-test {best_config['num_shots_test']} \
          --batch-size {best_config['batch']} \
          --num-steps {best_config['steps']} \
          --step-size {best_config['step_size']} \
          --meta-lr {best_config['lr']} \
          --num-epochs 150 \
          --num-batches {best_config['num_batches']} \
          --hidden-size {best_config['hidden']} \
          --output-folder {final_output_folder} \
          --verbose"""
        
        if self.use_cuda:
            cmd += ' --use-cuda'
            print(f"🎯 Flag GPU activé: --use-cuda")
        
        print(f"\n▶️  Lancement ENTRAÎNEMENT FINAL sur meta-train (150 epochs)...")
        print(f"Output folder: {final_output_folder}\n")
        
        original_dir = os.getcwd()
        os.chdir(self.base_dir)
        
        try:
            start_time = datetime.now()
            train_result = os.system(cmd)
            elapsed_time_train = (datetime.now() - start_time).total_seconds() / 60
            
            print(f"\n⏱️  Temps d'entraînement: {elapsed_time_train:.1f} minutes")
            
            if train_result != 0:
                print("❌ Entraînement final échoué!")
                return None
            
            # ✅ TEST sur meta-test (UNE SEULE FOIS!)
            config_path = os.path.join(final_output_folder, 'config.json')
            test_cmd = f"python test.py {config_path}"
            
            if self.use_cuda:
                test_cmd += ' --use-cuda'
            
            test_cmd += ' --verbose'
            
            print(f"\n▶️  Lancement TEST FINAL sur meta-test...")
            test_result = os.system(test_cmd)
            
            if test_result != 0:
                print("❌ Test final échoué!")
                return None
            
            # ✅ Charger résultats final du meta-test
            try:
                test_results_path = os.path.join(final_output_folder, 'test_results.json')
                with open(test_results_path, 'r') as f:
                    test_results = json.load(f)
                
                final_accuracy = test_results.get('accuracy', 0)
                final_loss = test_results.get('loss', 0)
                
                final_result = {
                    'final_accuracy': final_accuracy,
                    'final_loss': final_loss,
                    'elapsed_time_minutes': elapsed_time_train,
                    'config': best_config
                }
                
                self.all_results['final_test'] = {
                    'evaluation_set': 'meta-test',
                    'final_accuracy': final_accuracy,
                    'final_loss': final_loss,
                    'elapsed_time_minutes': elapsed_time_train,
                    'output_folder': final_output_folder,
                    'best_config': best_config
                }
                
                print(f"\n{'='*80}")
                print(f"🏆 RÉSULTAT FINAL - META-TEST")
                print(f"{'='*80}")
                print(f"✅ Test Accuracy: {final_accuracy:.2%}")
                print(f"✅ Test Loss: {final_loss:.4f}")
                print(f"⏱️  Temps entraînement: {elapsed_time_train:.1f} minutes")
                print(f"{'='*80}")
                
                return final_result
            
            except Exception as e:
                print(f"❌ Erreur lecture résultats: {e}")
                return None
        
        finally:
            os.chdir(original_dir)
    
    def generate_comparison_plots(self):
        """Générer graphiques de comparaison"""
        
        print("\n" + "="*80)
        print("📊 GÉNÉRATION DES GRAPHIQUES")
        print("="*80)
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Hyperparameter Tuning Results - OFAT (Val Set) | Final Test: {self.all_results.get("final_test", {}).get("final_accuracy", 0):.2%}', 
                         fontsize=16, fontweight='bold')
            
            # STAGE 1: Learning Rate
            if 'stage1' in self.all_results['stages']:
                stage1 = self.all_results['stages']['stage1']
                lrs = [r['lr'] for r in stage1['results']]
                accs = [r['accuracy'] for r in stage1['results']]
                
                axes[0, 0].plot(range(len(lrs)), accs, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
                axes[0, 0].set_xticks(range(len(lrs)))
                axes[0, 0].set_xticklabels([f'{lr:.4f}' for lr in lrs], rotation=45)
                axes[0, 0].set_ylabel('Accuracy (meta-val)', fontsize=12)
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
                axes[0, 1].set_ylabel('Accuracy (meta-val)', fontsize=12)
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
                axes[1, 0].set_ylabel('Accuracy (meta-val)', fontsize=12)
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
                axes[1, 1].set_ylabel('Accuracy (meta-val)', fontsize=12)
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
        """Afficher résumé final complet"""
        
        print("\n" + "="*80)
        print("🏆 RÉSUMÉ FINAL COMPLET - HYPERPARAMETERS OPTIMISÉS")
        print("="*80)
        
        if 'stage1' not in self.all_results['stages']:
            print("❌ Aucune donnée disponible!")
            return
        
        best_lr = self.all_results['stages']['stage1']['best']['lr']
        best_steps = self.all_results['stages']['stage2']['best']['steps'] if 'stage2' in self.all_results['stages'] else self.defaults['steps']
        best_batch = self.all_results['stages']['stage3']['best']['batch'] if 'stage3' in self.all_results['stages'] else self.defaults['batch']
        best_hidden = self.all_results['stages']['stage4']['best']['hidden'] if 'stage4' in self.all_results['stages'] else self.defaults['hidden']
        
        final_accuracy = self.all_results.get('final_test', {}).get('final_accuracy', 0)
        final_loss = self.all_results.get('final_test', {}).get('final_loss', 0)
        
        print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║ 🎯 MEILLEURS HYPERPARAMETERS TROUVÉS (META-VAL)                            ║
╚════════════════════════════════════════════════════════════════════════════╝

   • Learning Rate: {best_lr}
   • Num Steps: {best_steps}
   • Batch Size: {best_batch}
   • Hidden Size: {best_hidden}

╔════════════════════════════════════════════════════════════════════════════╗
║ 📊 ACCURACIES À CHAQUE ÉTAPE (ÉVALUATION SUR META-VAL)                    ║
╚════════════════════════════════════════════════════════════════════════════╝

   • ÉTAPE 1 (LR Tuning):     {self.all_results['stages']['stage1']['best']['accuracy']:.2%}
   • ÉTAPE 2 (STEPS Tuning):  {self.all_results['stages']['stage2']['best']['accuracy']:.2%}
   • ÉTAPE 3 (BATCH Tuning):  {self.all_results['stages']['stage3']['best']['accuracy']:.2%}
   • ÉTAPE 4 (HIDDEN Tuning): {self.all_results['stages']['stage4']['best']['accuracy']:.2%}

╔════════════════════════════════════════════════════════════════════════════╗
║ 🏆 RÉSULTAT FINAL (ÉVALUATION SUR META-TEST)                              ║
╚════════════════════════════════════════════════════════════════════════════╝

   • Final Test Accuracy: {final_accuracy:.2%} ✅
   • Final Test Loss: {final_loss:.4f}
   • Temps d'entraînement: {self.all_results.get('final_test', {}).get('elapsed_time_minutes', 0):.1f} minutes

╔════════════════════════════════════════════════════════════════════════════╗
║ 💻 DEVICE UTILISÉ                                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

   • Device: {self.device}
   • CUDA Available: {self.cuda_available}
   • GPU Model: {torch.cuda.get_device_name(0) if self.cuda_available else 'N/A'}

╔════════════════════════════════════════════════════════════════════════════╗
║ ✅ PIPELINE CORRECT APPLIQUÉ                                              ║
╚════════════════════════════════════════════════════════════════════════════╝

   TUNING (Étapes 1-4):
     ✅ Train sur meta-train
     ✅ Évaluer sur meta-val (pas meta-test!)
     ✅ Choisir meilleur paramètre

   TEST FINAL:
     ✅ Train complet sur meta-train (150 epochs)
     ✅ Évaluer sur meta-test UNE SEULE FOIS
     ✅ Résultat final JAMAIS VU AVANT!

╔════════════════════════════════════════════════════════════════════════════╗
║ 💡 PROCHAINES ÉTAPES                                                      ║
╚════════════════════════════════════════════════════════════════════════════╝

   1. Analyser le résultat final ({final_accuracy:.2%})
   2. Générer rapports et visualisations
   3. Sauvegarder le modèle final
   4. Documenter les résultats
        """)
    
    def run_complete_tuning(self):
        """Lancer le tuning complet + test final"""
        
        print("\n" + "="*80)
        print("🚀 DÉMARRAGE HYPERPARAMETER TUNING - STRATÉGIE OFAT")
        print("="*80)
        print(f"Timestamp: {self.all_results['timestamp']}")
        print(f"Résultats: {self.results_dir}")
        print(f"Device: {self.device}")
        print(f"CUDA Available: {self.cuda_available}")
        print(f"Pipeline: ✅ TRAIN/VAL/TEST CORRECT (pas de data leakage)")
        
        try:
            # ✅ ÉTAPE 1-4: TUNING (utilise meta-val)
            best_lr, results_lr = self.run_stage1_learning_rate()
            if best_lr is None:
                print("❌ Étape 1 échouée!")
                return None
            
            best_steps, results_steps = self.run_stage2_num_steps(best_lr)
            if best_steps is None:
                print("❌ Étape 2 échouée!")
                return None
            
            best_batch, results_batch = self.run_stage3_batch_size(best_lr, best_steps)
            if best_batch is None:
                print("❌ Étape 3 échouée!")
                return None
            
            best_hidden, results_hidden = self.run_stage4_hidden_size(best_lr, best_steps, best_batch)
            if best_hidden is None:
                print("❌ Étape 4 échouée!")
                return None
            
            # ✅ Assembler meilleure config
            best_config = {
                'lr': best_lr,
                'steps': best_steps,
                'batch': best_batch,
                'hidden': best_hidden,
                'num_ways': self.defaults['num_ways'],
                'num_shots': self.defaults['num_shots'],
                'num_shots_test': self.defaults['num_shots_test'],
                'step_size': self.defaults['step_size'],
                'num_batches': self.defaults['num_batches'],
                'expected_accuracy_on_val': self.all_results['stages']['stage4']['best']['accuracy']
            }
            
            # ✅ ÉTAPE FINALE: TEST sur meta-test UNE SEULE FOIS
            final_result = self.run_final_test(best_config)
            if final_result is None:
                print("❌ Test final échoué!")
                return None
            
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
            print("✅ TUNING COMPLET TERMINÉ!")
            print("="*80)
            
            return self.all_results
        
        except Exception as e:
            print(f"\n❌ ERREUR PENDANT TUNING: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    base_path = '.'
    
    if not os.path.exists(base_path):
        print(f"❌ Le dossier {base_path} n'existe pas!")
        exit(1)
    
    # ✅ Paramètre pour GPU
    use_cuda = True  # Mettre False pour forcer CPU
    
    # Créer tuner
    tuner = HyperparameterTuner(base_dir=base_path, use_cuda=use_cuda)
    
    # Lancer tuning complet + test final
    results = tuner.run_complete_tuning()
    
    if results:
        print("\n✅ Pipeline complet réussi!")
        print(f"📊 Accuracy final sur meta-test: {results.get('final_test', {}).get('final_accuracy', 0):.2%}")
    else:
        print("\n❌ Pipeline échoué!")