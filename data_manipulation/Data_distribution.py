import numpy as np
import random
import data_manipulation.Subset_dataset  as Subset_dataset
from tqdm import tqdm

class Data_distribution:
    
    def __init__(self, data_set, num_clients):
        """_summary_

        Args:
            data_set (_type_): the dataset we want to distribute
            num_clients (_type_): the number of clients of our FL system
        """
        self.data_set = data_set
        self.num_clients = num_clients
        
    def print_number_of_client(self):
        print(self.num_clients)
    
    def division_iid(self,dataset = None, num_clients = None, initial = "client"):
        """IID distribution of the data

        Args:
            initial (str, optional): The start of the name of each client. Defaults to "client".

        Returns:
            _type_: dictionary of the form {client_name, data index list)
        """
        if (dataset == None):
            dataset = self.data_set
            num_clients = self.num_clients
        
        # The division is IID where each client will receive the same amount of data 
        num_items = int(len(dataset)/num_clients)
    
        # Dict_user will contain a dictionary of the form {client_name, data index list)
        dict_users, all_idxs = {}, np.arange(len(dataset))
    
        # The name of the clients
        client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    
        for i in tqdm(range(num_clients)):
        
            idx_clienti = np.random.choice(all_idxs, num_items, replace=False)
            labels_clienti = [dataset[i][1] for i in idx_clienti]
        
            sub_data = Subset_dataset.Subset_dataset(dataset, idx_clienti,labels_clienti)
            dict_users[client_names[i]] = sub_data
            all_idxs = list(set(all_idxs) - set(idx_clienti))
    
        return dict_users
    
    def division_noniid_custom(self, num_classe_partage, num_intra_clients):
        """
                Extract I.I.D. client data from a dataset
                return: dict of client data
        """
    
        all_idxs = np.arange(len(self.data_set))
        dict_users, all_idxs = dict(), np.arange(len(self.data_set))
        labels = np.array(self.data_set.targets)
    
        idxs_labels = np.vstack((all_idxs, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
        unique_labels = np.unique(np.array(labels))
        # on mélange les labels pour que le résultat finale ne soit pas biasé
        random.shuffle(unique_labels)

        all_idxs = idxs_labels[0]
        labels = idxs_labels[1]
    
        sub_lab_list = [unique_labels[i:i + num_classe_partage] for i in range(0, len(unique_labels), num_classe_partage)]
   
        for item in tqdm.notebook.tqdm(sub_lab_list):
        
            idx_clienti = np.extract([labels[i] in item for i in range(len(labels))], all_idxs)
            labels_clienti = [self.data_set[i][1] for i in idx_clienti]
        
        
            # Creation du nom de chaque client
            initial = 'client'
            for lab in item:
                initial = initial + str(lab) + '_'
                    
            sub_data = Subset_dataset.Subset_dataset(self.data_set, idx_clienti,labels_clienti)
            # Pour les clients intraclass la division sera iid donc on utilisera la fonction qu'on a déja implementé et on utilise aussi la classe subset_dataset
            intraclass_clients = self.division_iid(dataset = sub_data, num_clients = num_intra_clients, initial = initial)

            dict_users.update(intraclass_clients)
    
        return dict_users
    
    def division_noniid_fedlab(self, indx_clients, indx_labels):
        """
                Extract I.I.D. client data from a dataset
                return: dict of client data
        """
        
        dict_users, all_idxs = dict(), np.arange(len(self.data_set))

        num_client = 0
        for index_client in tqdm(indx_clients):
            
            labels_clienti = [self.data_set[i][1] for i in index_client]
            
            # Creation du nom de chaque client
            name_client = 'client'
            for lab in indx_labels[num_client]:
                name_client = name_client + '_' + str(lab)  
            name_client = name_client + '_' + str(num_client)        

            sub_data = Subset_dataset.Subset_dataset(self.data_set, index_client,labels_clienti)
            
            dict_users[name_client] =  sub_data
            num_client = num_client + 1
    
        return dict_users
    
    
