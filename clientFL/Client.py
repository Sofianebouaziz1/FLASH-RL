import torch
from torch.utils.data import DataLoader
import copy
from torch.nn import functional as F
import time

class Client(object):
    
    def __init__(self, client_name, local_dataset, local_model, numbercores , frequency, bandwidth, loss, B, learning_rate, momentum):
        """
        Initializes a client for FL.

        Parameters:
        - client_name (str): Name of the client.
        - local_dataset: The local dataset for the client.
        - local_model: The local model for the client.
        - numbercores (int): Number of cores available on the client.
        - frequency: Distribution of frequency.
        - bandwidth: Distribution of bandwidth.
        - loss: Loss function used for training.
        - B (int): Size of a batch.
        - learning_rat (float) e: Learning rate for the optimizer.
        - momentum (float): Momentum for the optimizer.
        """
        
        
        # name of the client
        self.client_name = client_name
        # number of cores
        self.numbercores = numbercores 
        # the distribution of the frequency
        self.frequency = frequency
        # the distribution of the bandwidth
        self.bandwidth =  bandwidth
        # size of a batch
        self.B = B
        # local data recieved from server 
        self.local_data = DataLoader(local_dataset, batch_size= B, shuffle=True)
        # local model
        self.local_model = copy.deepcopy(local_model)
        # loss function we are using
        self.loss_func = loss
        # the optimizer
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr= learning_rate , momentum = momentum)


        
    def set_weights(self, global_parameters):
        """
        Set the weights of the local model using the global model's parameters.

        Parameters:
        - global_parameters: The global model parameters recieved from the server.
        """
        self.local_model.load_state_dict(global_parameters)

        
    def train(self, global_parameters, E, mu, type_data, verbos=0):
        """
        Train the local model based on the specified type of data.

        Parameters:
        - global_parameters: The global model parameters.
        - E (int): Local number of epoch.
        - mu: FedProx parameter.
        - type_data (str): Type of data, e.g., "Fall" or other.
        - verbos (int): Verbosity level for printing training information (default is 0).
        
        Returns:
        - The result of the training.
        """
        if type_data == "Fall":
            return self.train_falldetection(global_parameters, E, mu, verbos)
        else:
            return self.train_others(global_parameters, E, mu, verbos)

        

    def train_falldetection(self, global_parameters, E, mu, verbos):
        """
        Train the local model for fall detection.

        Parameters:
        - global_parameters: The global model parameters.
        - E (int): Local number of epoch.
        - mu: FedProx parameter.
        - verbos (int): Verbosity level for printing training information.

        Returns:
        - A deep copy of the state dictionary of the trained local model.
        """


        # Initialize local model parameters by global_parameters
        self.local_model.load_state_dict(global_parameters)
        
        # Start local training
        self.local_model.train()

        for iter in range(E):
            if (verbos == 1) :
                print("Client : ",self.client_name," Iteration :",iter+1)
                
            for idx, (X_batch, y_batch) in enumerate(self.local_data):
                # Initialize the gradients to 0
                self.local_model.zero_grad()
                out = self.local_model(X_batch.float())
                loss = self.loss_func(out, y_batch)
                # The addition of the term proximal

                if (mu != 0 and iter > 0):
                    for w, w_t in zip(self.local_model.parameters(), global_parameters.values()):
                        loss += mu / 2. * torch.pow(torch.norm(w.data - w_t.data), 2)

                # Calculation of gradients
                loss.backward()
                
                # Update of the parameters
                self.optimizer.step()


        return copy.deepcopy(self.local_model.state_dict())
         
    def train_others(self, global_parameters, E, mu, verbos):
        """
        Train the local model for a task other than fall detection.

        Parameters:
        - global_parameters: The global model parameters.
        - E (int): Local number of epoch.
        - mu: FedProx parameter.
        - verbos (int): Verbosity level for printing training information.

        Returns:
        - A deep copy of the state dictionary of the trained local model.
        """
        
        # Initialize local model parameters by global_parameters
        self.local_model.load_state_dict(global_parameters)
        
        # Start local training
        self.local_model.train()

        for iter in range(E):
            if (verbos == 1) :
                print("Client : ",self.client_name," Iteration :",iter+1)
            
            for images, labels in self.local_data:
                # Initialize the gradients to 0
                self.local_model.zero_grad()
                # Probability calculation for batch i images
                log_probs = self.local_model(images)
                # Loss calculation
                loss = self.loss_func(log_probs, labels)
                
                # The addition of the term proximal
                if (mu != 0 and iter > 0):
                        for w, w_t in zip(self.local_model.parameters(), global_parameters.values()):
                            loss += mu / 2. * torch.pow(torch.norm(w.data - w_t.data), 2)
                
                # Calculation of gradients
                loss.backward()
                
                # Update of the parameters
                self.optimizer.step()
                
        return copy.deepcopy(self.local_model.state_dict())


    def get_size(self):
        """
        Get the size of the local dataset.

        Returns:
        - The size of the local dataset.
        """
        return len(self.local_data)

    def get_model(self):
        """
        Get the local model.

        Returns:
        - The local model.
        """
        return self.local_model

    def delete_model(self):
        """
        Delete the local model to free up resources.
        """
        del self.local_model
