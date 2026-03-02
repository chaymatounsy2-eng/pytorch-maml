"""
Wrapper pour ThermalMetaDataset
Transforme le format dict en format attendu par MAML
"""

class ThermalBatchIterator:
    """
    Itérateur pour batches thermiques
    
    Transforme batch dict {'train': (x, y), 'test': (x, y)}
    En liste de dicts attendus par MAML
    """
    
    def __init__(self, dataloader, batch_size):
        """
        Arguments:
        -----------
        dataloader: torch.utils.data.DataLoader
            DataLoader standard pour ThermalMetaDataset
        
        batch_size: int
            Nombre de tâches par batch
        """
        self.dataloader = dataloader
        self.batch_size = batch_size
    
    def __iter__(self):
        """
        Itère sur les batches et les divise en tâches individuelles
        """
        for batch in self.dataloader:
            # batch = {'train': (train_x, train_y), 'test': (test_x, test_y)}
            # où train_x shape = (batch_size*10, 3, 224, 224)
            #     train_y shape = (batch_size*10,)
            #     test_x shape = (batch_size*20, 3, 224, 224)
            #     test_y shape = (batch_size*20,)
            
            train_x, train_y = batch['train']
            test_x, test_y = batch['test']
            
            # Diviser en tâches individuelles
            # Chaque tâche a:
            # - 10 images en train (5 healthy + 5 diseased)
            # - 20 images en test (10 healthy + 10 diseased)
            tasks = []
            images_per_task_train = 10
            images_per_task_test = 20
            
            for i in range(self.batch_size):
                # Indices pour cette tâche
                start_train = i * images_per_task_train
                end_train = (i + 1) * images_per_task_train
                start_test = i * images_per_task_test
                end_test = (i + 1) * images_per_task_test
                
                # Créer la tâche au format attendu par MAML
                # MAML.get_outer_loss() cherche batch[i]['train'] et batch[i]['test']
                task = {
                    'train': (train_x[start_train:end_train],      # train_inputs
                              train_y[start_train:end_train]),     # train_targets
                    'test': (test_x[start_test:end_test],          # test_inputs
                             test_y[start_test:end_test])          # test_targets
                }
                tasks.append(task)
            
            # Retourner les tâches au format attendu par MAML
            # MAML.train_iter itère sur le batch et appelle enumerate(batch)
            yield tasks