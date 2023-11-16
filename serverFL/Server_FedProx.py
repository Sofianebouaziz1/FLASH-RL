import copy
import clientFL.Client as Client
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,recall_score,precision_score
from torch.nn import functional as F
import timeit

class Server_FedProx(object):
    
    
    
    def __init__(self, num_clients, global_model, dict_clients, loss_fct, B, dataset_test, learning_rate, momentum, clients_info):
        """
        Initialize the Server_FedProx object.

        Parameters:
        - num_clients (int): The number of clients in the system.
        - global_model: The global model (our goal).
        - dict_clients (dict): A dictionary containing the information about each client.
        - loss_fct: The loss function used for training.
        - B (int): The size of the batch.
        - dataset_test: The test dataset used to evaluate each round.
        - learning_rate: The learning rate for the clients.
        - momentum: The momentum for the clients.
        - clients_info: Information about clients for simulation.
        """
   
        # The number of clients in the system
        self.N = num_clients
        # The global model (our goal)
        self.model = global_model 
        # A list containing the @ of each client
        self.list_clients =  [] 
        # The size of the batch 
        self.B = B
        # The test dataset that help us to evaluates each round
        self.dataset_test = dataset_test
        # Test Dataloader
        self.testdataloader = DataLoader(self.dataset_test, batch_size= self.B)
        # The distribution of clients in a dict
        self.dict_clients = dict_clients
        # The loss function
        self.loss_function = copy.deepcopy(loss_fct)
        # The number of parameters
        self.number_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Call a function to create the clients (simulation)
        self.clients_info = clients_info
        self.create_clients(learning_rate, momentum)
   
    def create_clients(self, learning_rate, momentum):
        '''
        Create client objects based on the provided information and add them to the list.

        Parameters:
        - learning_rate: The learning rate for training clients.
        - momentum: The momentum parameter for training clients.
        '''
        cpt = 0
        for client_name in self.dict_clients.keys():
            # Create a client object with the specified parameters
            client = Client.Client(
                self.clients_info[cpt][0],  # client name
                self.dict_clients[client_name],  # client information
                copy.deepcopy(self.model),  # copy of the global model
                self.clients_info[cpt][2],  # number of cores
                self.clients_info[cpt][3],  # frequency distribution
                self.clients_info[cpt][4],  # bandwidth distribution
                copy.deepcopy(self.loss_function),  # copy of the loss function
                self.B,  # size of the batch
                learning_rate,  # learning rate
                momentum  # momentum
            )
            # Append the client to the list
            self.list_clients.append(client)
            cpt += 1


    def weight_scalling_factor(self, client, active_clients):
        '''
        Determine the factor assigned to a given client.

        Parameters:
        - client: The client for which the factor is determined.
        - active_clients: List of active clients.

        Returns:
        - The scaling factor for the client.
        '''
        # Calculate the total training data points across clients for this round
        global_count = sum([client_obj.get_size() for client_obj in active_clients])
        # Get the total number of data points held by a client
        local_count = client.get_size()

        return local_count / global_count

    def scale_model_weights(self, weight, scalar):
        '''
        Multiply the local parameters of each client by its factor.

        Parameters:
        - weight: The local model weights.
        - scalar: The scaling factor.

        Returns:
        - Scaled local model weights.
        '''
        w_scaled = copy.deepcopy(weight)

        for k in weight.keys():
            w_scaled[k] = scalar * w_scaled[k]

        return w_scaled

    def sum_scaled_weights(self, scaled_weight_list):
        '''
        Aggregate different parameters.

        Parameters:
        - scaled_weight_list: List of scaled models weights.

        Returns:
        - Aggregated weights.
        '''
        w_avg = copy.deepcopy(scaled_weight_list[0])

        for k in w_avg.keys():
            tmp = torch.zeros_like(scaled_weight_list[0][k], dtype=torch.float32)

            for i in range(len(scaled_weight_list)):
                tmp += scaled_weight_list[i][k]

            w_avg[k].copy_(tmp)

        return w_avg
            
    def select_active_clients(self, comm_round, C, drop_percent=0):
        '''
        Select a fraction of active clients for a training round.

        Parameters:
        - comm_round (int): The current communication round.
        - C (float): The fraction of active clients.
        - drop_percent (float): The percentage of clients to drop (optional, default is 0).

        Returns:
        - active_clients (list): List of active client objects for the current round.
        '''
        # max between c*k and 1
        client_index = np.arange(0, len(self.list_clients))
        
        m = int(max(C * self.N, 1))  # max between c*k and 1
        active_clients_index = random.sample(list(client_index), k=m)  # Select a fraction of clients
        active_clients = [self.list_clients[i] for i in active_clients_index]
        # print(active_clients_index)

        return active_clients

        def flatten(self, weight):
        '''
        Flatten the parameters of a model.

        Parameters:
        - weight: The model weights.

        Returns:
        - Flattened model weights.
        '''
        weight_flatten = []

        for param in weight.values():
            weight_flatten.append(np.array(param).reshape(-1))

        weight_flatten = [item for sublist in weight_flatten for item in sublist]

        return weight_flatten

    def global_train(self, comms_round, C, E, mu, verbose_test=1, verbos=0, type_data="others", init_weights=None, init_commround=None):
        '''
        Train the global model using the Federated Learning approach.

        Parameters:
        - comms_round (int): The number of communication rounds.
        - C (float): The fraction of active clients in each round.
        - E (int): The number of local epochs for each client in each round.
        - mu (float): The learning rate for local training.
        - verbose_test (int, optional): Verbosity level for test results (default is 1).
        - verbos (int, optional): Verbosity level for training details (default is 0).
        - type_data (str, optional): Type of data for training and testing ("others" or "Fall", default is "others").
        - init_weights (dict, optional): Initial weights for the global model (default is None).
        - init_commround (int, optional): Initial communication round (default is None).

        Returns:
        - dict_result (dict): Dictionary containing training results, including accuracy, loss, and other metrics.
        '''
        if type_data == "Fall":
            return self.global_train_fall(comms_round, C, E, mu, verbose_test, verbos, type_data, init_weights, init_commround)
        else:
            return self.global_train_others(comms_round, C, E, mu, verbose_test, verbos, type_data, init_weights, init_commround)

    
    def global_train_others(self, comms_round, C, E, mu, verbose_test = 1, verbos = 0, type_data = "others",  init_weights = None, init_commround = None):

        
        rounds = []
        accuarcy = []
        loss = []

        time_rounds = []
        time_rounds_sum = []
        best_model_weights = {}
        best_accuracy = 0

        max_latency = 0
        min_latency = 100000000
        
        for client in self.list_clients:
            frequency_client = random.choice(client.frequency)
            bandwidth_client = random.choice(client.bandwidth)
            
            latency_min = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * max(client.frequency)) + (self.number_parameters * 64)/(1000000 * max(client.bandwidth))
            latency_max = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * min(client.frequency)) + (self.number_parameters * 64)/(1000000 * min(client.bandwidth))

            if (latency_max > max_latency ):
                max_latency = latency_max 
                
            if (latency_min < min_latency):
                min_latency = latency_min

        if (init_weights != None):
            self.model.load_state_dict(copy.deepcopy(init_weights))
            
            comm_start = init_commround
                      
            for client in self.list_clients :
                client.set_weights(copy.deepcopy(init_weights))
            
        else:
            comm_start =  0
            
        self.model.train()
             
        # For each training round
        for comm_round in  range(comm_start, comms_round):

            temps_debut = timeit.default_timer()
            rounds.append(comm_round+1)
            
            if (verbos == 1):
                print("*************************************************************************************")
                print("Communication round n : ", comm_round + 1)

            # Receive the parameters of the global model (first we will have the initialized parameters)
            global_weights = self.model.state_dict()
            
            
            # List to collect the parameters of the model * weight of each client
            scaled_local_weight_list = []

            # Select a client portion C
            active_clients = self.select_active_clients(comm_round, C)

            time_roundt = []
            # For each selected customer
            for client in active_clients:
                if (verbos == 1):
                        print("Entrainnement locale du client : ", client.client_name)

                # Training on local client data
                client_w = client.train(copy.deepcopy(global_weights), E, mu, type_data, verbos)

                # Get the parameters of the local model * factor
                client_scaling_factor = self.weight_scalling_factor(client, active_clients) #nk/n'
                    
                # weights*factor
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor) 
                
                # save in the list
                scaled_local_weight_list.append(client_scaled_weight) 
                
                                
                frequency_client = random.choice(client.frequency)
                bandwidth_client = random.choice(client.bandwidth)
                
                latency_client = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * frequency_client) + (self.number_parameters * 64)/(1000000 * bandwidth_client)
            
                #print("Client name : ", client.client_name, " with latency :", latency_client)
                time_roundt.append(latency_client)

            time_rounds.append(max(time_roundt))
            time_rounds_sum.append(sum(time_roundt))
            
            # The average of the different points received
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)

            # Copy the weights in the global model
            self.model.load_state_dict(average_weights)

            acc_test, loss_test = self.test(type_data)
            
            # Tester le modele gloable chaque round
            if (verbose_test == 1):
                print("Training round n :", (comm_round+1),", Test accuarcy : ", round(acc_test.numpy()*100, 2) ,", Test loss :", round(loss_test, 2))
                #print("Max time : ", max(time_roundt))
                print("*************************************************************************************")
            
            if (acc_test > best_accuracy):
                best_accuracy = acc_test
                best_model_weights = copy.deepcopy(average_weights)
                    
            accuarcy.append(acc_test.numpy())
            loss.append(loss_test)
            
            temps_fin = timeit.default_timer() - temps_debut
            #time_rounds.append(temps_fin)
         
        for client in self.list_clients:
            client.delete_model()     
            
        dict_result = {
            "Best_model_weights": best_model_weights,
            "Accuracy": accuarcy,
            "Loss" : loss,
            "Timeurounds" : time_rounds,
            "Timesum" : time_rounds_sum
        }
            
        return dict_result
 
    
    def global_train_fall(self, comms_round, C, E, mu, verbose_test = 1, verbos = 0, type_data = "others",  init_weights = None, init_commround = None):
        '''Global model training'''
        
        rounds = []
        accuarcy = []
        recall = []
        precision = []
        loss = []
        f1score = []
        
        time_rounds = []
        time_rounds_sum = []
        best_model_weights = {}
        best_f1score = 0
        
        if (init_weights != None):
            self.model.load_state_dict(copy.deepcopy(init_weights))
            
            comm_start = init_commround
                      
            for client in self.list_clients :
                client.set_weights(copy.deepcopy(init_weights))
            
        else:
            comm_start =  0
            
        client_selected = []
        self.model.train()

        # For each training round
        for comm_round in  tqdm(range(comm_start, comms_round)):

            rounds.append(comm_round+1)
            
            if (verbos == 1):
                print("*************************************************************************************")
                print("Communication round n : ", comm_round + 1)

            # Receive the parameters of the global model (first we will have the initialized parameters)
            global_weights = self.model.state_dict()

            
            # List to collect the parameters of the model * weight of each client
            scaled_local_weight_list = []

            # Select a client portion C
            active_clients = self.select_active_clients(comm_round, C)

            time_roundt = []
            
            client_selected_tour = []
            
            for client in active_clients:
                client_selected_tour.append(client.client_name)
            
            client_selected.append(client_selected_tour)
            # For each selected customer
            for client in active_clients:
                if (verbos == 1):
                     print("Entrainnement locale du client : ", client.client_name)

                # Training on local client data
                client_w = client.train(copy.deepcopy(global_weights), E, mu, type_data, verbos)

                # Get the parameters of the local model * factor
                client_scaling_factor = self.weight_scalling_factor(client, active_clients) #nk/n'
                    
                # weights*factor
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor) 
                
                # save in the list
                scaled_local_weight_list.append(client_scaled_weight) 
                
                frequency_client = random.choice(client.frequency)
                bandwidth_client = random.choice(client.bandwidth)
                
                latency_client = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * frequency_client) + (self.number_parameters * 64)/(1000000 * bandwidth_client)
            
                #print("Client name : ", client.client_name, " with latency :", latency_client)
                time_roundt.append(latency_client)

            time_rounds.append(max(time_roundt))
            time_rounds_sum.append(sum(time_roundt))
            
            # The average of the different points received
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)

            # Copy the weights in the global model
            self.model.load_state_dict(average_weights)

            acc_test, f1score_test, recall_test, precision_test, loss_test = self.test(type_data)
        
            # Test the global model each round
            if (verbose_test == 1):
                print("Training round n :", (comm_round+1),", Test accuarcy : ", round(acc_test*100, 2) ,", Test F1_score :", round(f1score_test*100, 2)," Test loss :", round(loss_test, 2))
                print("Test Recall : ", round(recall_test*100, 2), "Test Precision : ", round(precision_test*100, 2))
                print("Max time : ", max(time_roundt))
                print("*************************************************************************************")
                
            if (f1score_test > best_f1score):
                best_f1score = f1score_test
                best_model_weights = copy.deepcopy(average_weights)
                    
            accuarcy.append(acc_test)
            f1score.append(f1score_test)
            recall.append(recall_test)
            precision.append(precision_test)
            loss.append(loss_test)
        
        for client in self.list_clients:
            client.delete_model()  
                
        dict_result = {
            "Best_model_weights": best_model_weights,
            "Accuracy": accuarcy,
            "F1_score": f1score,
            "Recall" : recall,
            "Precision" : precision,
            "Loss" : loss,
            "Timeurounds" : time_rounds,
            "Timesum" : time_rounds_sum,
            "Client_selected" : client_selected
        }
            
        return dict_result


   def test(self, type_data):
        """
        Test the global model on the specified type of data.

        Parameters:
        - type_data: Type of data, e.g., "Fall" or "others".

        Returns:
        - The test accuracy and loss.
        """
        if type_data == "Fall":
            return self.test_falldetection()
        else:
            return self.test_others()

    
    def test_others(self):
        """
        Evaluate the global model with the test dataset for a task other than fall detection.

        Returns:
        - The accuracy and loss on the test dataset.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Testing
        test_loss = 0
        correct = 0

        # Iterate through the test dataset
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.testdataloader):
                log_probs = self.model(data)
                # Sum up batch loss
                test_loss += torch.nn.functional.cross_entropy(log_probs, target, reduction='sum').item()

                # Get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            # Calculate average test loss and accuracy
            test_loss /= len(self.testdataloader.dataset)
            accuracy = correct / len(self.testdataloader.dataset)

        return accuracy, test_loss

    
    def test_falldetection(self):
        """
        Evaluate the global model on a test dataset for fall detection.

        Returns:
        - Accuracy, F1 score, recall, precision, and epoch loss on the fall detection test dataset.
        """
        epoch_loss = 0
        correct, total = 0, 0
        targets = list()
        preds = list()
        acc = 0

        # Set the model to evaluation mode
        self.model.eval()

        for idx, (X_batch, y_batch) in enumerate(self.testdataloader):
            batchsize = X_batch.shape[0]

            # Forward pass
            out = self.model(X_batch.float())
            loss = self.loss_function(out.contiguous(), y_batch.contiguous())

            # Predictions and metrics
            pred = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (pred == y_batch).sum().item()

            targets += list(y_batch)
            preds += list(pred)

            epoch_loss += (loss.detach().item() / batchsize)

        # Calculate accuracy and additional metrics
        acc = correct / total
        f1score = f1_score(targets, preds, zero_division=1)
        recall = recall_score(targets, preds, zero_division=1)
        precision = precision_score(targets, preds, zero_division=1)

        return acc, f1score, recall, precision, epoch_loss
