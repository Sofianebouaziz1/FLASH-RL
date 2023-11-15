import collections
import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from scipy import signal
import os
import random
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

class MobiAct_divisions(object):
    
    
    def __init__(self, annoted_path):
        
        self.annoted_path = annoted_path # Path to the annoted data (data/MobiAct_Dataset_v2.0/Annotated Data for example)
        
        
    def resample_data(self):
        
        """
        This function has been taken from the following github repository :
            git@github.com:IKKIM00/fall-detection-and-predction-using-GRU-and-LSTM-with-Transfer-Learning.git    
        """
        fall_types = ['FOL', 'FKL', 'BSC', 'SDL']
        adl_types = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO', 'LYI']
        person_numlist = list(range(1, 68))
        trials = list(range(1, 4))
        os.makedirs("data_resampled/Mobidata_resampled/fall")
        os.makedirs("data_resampled/Mobidata_resampled/adl")

        """
        Resampling Fall Data
        """
        print("Resampling Fall Data ...")
        for person_num in tqdm(person_numlist):
            for fall_type in fall_types:
                for trial in trials:
                    try:
                        directory = self.annoted_path + "/"+ fall_type+ "/" + fall_type  + '_' + str(person_num) + '_' + str(trial) + '_annotated.csv'
                        data = pd.read_csv(directory)

                        std_length = len(data[data['label']=='STD'])
                        fall_length = len(data[data['label']==fall_type])
                        #lyi_length = len(data[data['label']=='LYI'])
                        
                        # Extract only the data of type fall
                        fall_data = data[std_length : std_length + fall_length]
                        
                        # Take the acc_x, acc_y and acc_z
                        df = fall_data[['acc_x', 'acc_y', 'acc_z']]

                        # The sizes of the scaffolding
                        n = 10
                        resampling_size = 30
                        
                        # If length of STD is higher than 10
                        if std_length >= n:
                            # 10 samples from STD + FALL resampled into 30 samples + 10 samples from LYI
                            # acc_x
                            std_x = pd.DataFrame(data['acc_x'][std_length - n : std_length]) # 10 STD
                            fall_x = pd.DataFrame(signal.resample(df['acc_x'], resampling_size)) #Fall data
                            fall_x.columns = ['acc_x']
                            lyi_x = pd.DataFrame(data['acc_x'][fall_length : fall_length + n]) # 10 LYI
                            result = pd.concat([std_x, fall_x, lyi_x], axis=0) # Concaténation
                            
                            # acc_y
                            std_y = pd.DataFrame(data['acc_y'][std_length - n : std_length])
                            fall_y = pd.DataFrame(signal.resample(df['acc_y'], resampling_size))
                            fall_y.columns = ['acc_y']
                            lyi_y = pd.DataFrame(data['acc_y'][fall_length : fall_length + n])
                            result['acc_y'] = pd.concat([std_y, fall_y, lyi_y])
                            
                            # acc_z
                            std_z = pd.DataFrame(data['acc_z'][std_length - n : std_length])
                            fall_z = pd.DataFrame(signal.resample(df['acc_z'], resampling_size))
                            fall_z.columns = ['acc_z']
                            lyi_z = pd.DataFrame(data['acc_z'][fall_length : fall_length + n])
                            result['acc_z'] = pd.concat([std_z, fall_z, lyi_z])
                            
                            # newly indexing
                            result.index = list(range(0, len(result)))

                            result['label'] = 0   #initialzing for label
                            result.loc[n: n + resampling_size - 1, 'label'] = 1    # labelling as 'FALL' = 1 les données de chutes
                        
                        # If the length of STD is shorter than 10
                        else:
                            # STD + FALL resampled into 30 samples + 10 samples from LYI
                            # acc_x
                            std_x = pd.DataFrame(data['acc_x'][: std_length]) # std_length STD
                            fall_x = pd.DataFrame(signal.resample(df['acc_x'], resampling_size)) # chutes
                            fall_x.columns = ['acc_x']
                            lyi_x = pd.DataFrame(data['acc_x'][fall_length : fall_length + n + (n - std_length)]) # n + la difference de n - std_length données de LYI
                            result = pd.concat([std_x, fall_x, lyi_x], axis=0)
                            
                            #acc_y
                            std_y = pd.DataFrame(data['acc_y'][: std_length])
                            fall_y = pd.DataFrame(signal.resample(df['acc_y'], resampling_size))
                            fall_y.columns = ['acc_y']
                            lyi_y = pd.DataFrame(data['acc_y'][fall_length : fall_length + n + (n - std_length)])
                            result['acc_y'] = pd.concat([std_y, fall_y, lyi_y])
                            
                            #acc_z
                            std_z = pd.DataFrame(data['acc_z'][: std_length])
                            fall_z = pd.DataFrame(signal.resample(df['acc_z'], resampling_size))
                            fall_z.columns = ['acc_z']
                            lyi_z = pd.DataFrame(data['acc_z'][fall_length : fall_length + n + (n - std_length)])
                            result['acc_z'] = pd.concat([std_z, fall_z, lyi_z])
                            
                            #newly indexing
                            result.index = list(range(0, len(result)))
                            
                            result['label'] = 0   #initialzing for label -> ADL
                            result.loc[std_length:std_length + resampling_size - 1, 'label'] = 1    # labelling as 'FALL' = 1
                            
                        # saving resampled data
                        save_route = "data_resampled/Mobidata_resampled/fall/" + fall_type + '_' + str(person_num) + '_' + str(trial) +'.csv'
                        result.to_csv(save_route)
                    except FileNotFoundError:
                        print(fall_type, person_num, trial, 'no data')
                        
                        
        """
        Resampling ADL Data
        """
        print("Resampling ADL Data ...")
        for person_num in tqdm(person_numlist):
            for adl_type in adl_types:
                for trial in trials:
                    try:
                        directory = self.annoted_path + "/" + adl_type+ "/" + adl_type  + '_' + str(person_num) + '_' + str(trial) + '_annotated.csv'
                        
                        data = pd.read_csv(directory)
                        
                        length = len(data) // 4
                        
                        # acc_x
                        result = pd.DataFrame(signal.resample(data['acc_x'][:length], 50))
                        result.columns = ['acc_x']
                        
                        # acc_y
                        result['acc_y'] = pd.DataFrame(signal.resample(data['acc_y'][:length], 50))
                        
                        # acc_z
                        result['acc_z'] = pd.DataFrame(signal.resample(data['acc_z'][:length], 50))
                        
                        # reindexing
                        result.index = list(range(0, 50))
                        
                        result['label'] = 0 # ADL
        
                        # saving resampled data
                        save_route = "data_resampled/Mobidata_resampled/adl/" + adl_type + '_' + str(person_num) + '_' + str(trial) +'.csv'
                        result.to_csv(save_route)
                    except:
                        print(adl_type, person_num, trial, 'no data')
                        
                        
    def create_users_folders(self):
        
        for num_user in tqdm(range(1,68)):
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user))
            
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/Fall")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/Fall/FOL") 
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/Fall/FKL") 
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/Fall/BSC") 
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/Fall/SDL") 
            
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL")
            os.makedirs("data_division/MobiAct//Users/User"+str(num_user)+"/ADL/STD")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/WAL")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/JOG")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/JUM")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/STU")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/STN")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/SCH")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/SIT")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/CHU")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/CSI")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/CSO")
            os.makedirs("data_division/MobiAct/Users/User"+str(num_user)+"/ADL/LYI")
        
            
    def populate_users(self, window):
    
        fall_data_dir = "data_resampled/Mobidata_resampled/fall/*.csv"
        adl_data_dir = "data_resampled/Mobidata_resampled/adl/*.csv"

        n = window

        print("Populate Fall types ...")
        nb_fall_timeseries = 0
        for file in tqdm(glob.glob(fall_data_dir)):
            
            data = pd.read_csv(file)

            folder = os.path.basename(file)
            fall_type = folder.split("_")[0] # Get the type of the Fall
            person_num = folder.split("_")[1] # Get the number of the person
            trial = folder.split("_")[2] # Get the number of the trial

            for i in range(0, len(data) - n):
                
                temp = list()
                """
                For each window of size = window, we will create a csv file that is equal to the size of the window, 
                i.e. each line in the file, will contain 4 components, the value of the X-axis of the accelerometer, the value of the Y-axis, Z-axis and the label 
                of this sample (for the Fall, it is either STD, Fall type or LYI).
                """ 
                for x, y, z, label in zip(data['acc_x'][i : i + n], data['acc_y'][i : i + n], data['acc_z'][i : i + n], data['label'][i: i + n]):
                    temp.append([x]+[y]+[z]+[label])
     
                nb_fall_timeseries = nb_fall_timeseries + 1
                
                example_df = pd.DataFrame(temp) # Transform it to pd in order to be saved as csv file
                example_df.columns = ["Accelerometer_x", "Accelerometer_y", "Accelerometer_z", "Label"] # rename the columns
                
                save_route = "data_division/MobiAct/Users/User"+str(person_num)+"/Fall/"+str(fall_type)+"/trail_"+str(trial)+"_timeserie"+str(i)+".csv"
                example_df.to_csv(save_route)
            
                
        print("Populate ADL ...")
        nb_adl_timeseries = 0
        for file in tqdm(glob.glob(adl_data_dir)):
            
            data = pd.read_csv(file)
            
            folder = os.path.basename(file)
            ADL_type = folder.split("_")[0]
            person_num = folder.split("_")[1]
            trial = folder.split("_")[2]
            
            for i in range(0, len(data) - n):
                temp = list()
                for x, y, z, label in zip(data['acc_x'][i : i + n], data['acc_y'][i : i + n], data['acc_z'][i : i + n], data['label'][i: i + n]):
                    temp.append([x]+[y]+[z]+[label])
                    
                nb_adl_timeseries = nb_adl_timeseries + 1

                example_df = pd.DataFrame(temp)
                example_df.columns = ["Accelerometer_x", "Accelerometer_y", "Accelerometer_z", "Label"]
                
                save_route = "data_division/MobiAct/Users/User"+str(person_num)+"/ADL/"+str(ADL_type)+"/trail_"+str(trial)+"_timeserie"+str(i)+".csv"
                example_df.to_csv(save_route)
 
        print("In total we have ", nb_fall_timeseries, "Time series of type Fall and ", nb_adl_timeseries, " Time series of type ADL") 
        
    def create_test_dataloader(self, seed, number_client_test):    
        """"
        We will select at random number_client_test clients and concatenate there time series to form a test centralized dataset. 
        We believe this is the best way to reflect a test dataset in reality.
        """   
        random.seed(seed)
        self.test_clients = random.sample(range(1, 68), number_client_test)
        self.FL_clinets = set(range(1,68)) - set(self.test_clients)
    
        
        X_test_fall = []
        Y_test_fall = []

        X_test_adl = []
        Y_test_adl = []
        x = 0
        for client_num in tqdm(self.test_clients):
            
            root_dir_fall = "data_division/MobiAct/Users/User"+str(client_num)+"/Fall"
            
            for subdir in os.listdir(root_dir_fall):
                subdir_path = os.path.join(root_dir_fall, subdir) # We will receive the types of "Falls" in this step
                for subdir2 in os.listdir(subdir_path): 
                    subdir2_path = os.path.join(subdir_path, subdir2) # We will receive the CSV files
                    
                    data = pd.read_csv(subdir2_path) # Read the time serie
                    
                    # See which label we will assign to the time series
                    cnt = collections.Counter(data['Label'])
                    num_adl = cnt[0]
                    num_fall = cnt[1]
                    
                    if num_fall == 30:
                        x = x + 1
                        X_test_fall.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']])) 
                        Y_test_fall.append(1) # 1 : fall
                    else :
                        X_test_adl.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))
                        Y_test_adl.append(0) # 0 : not fall
                        
            root_dir_adl = "data_division/MobiAct/Users/User"+str(client_num)+"/ADL"
            
            for subdir in os.listdir(root_dir_adl):
                subdir_path = os.path.join(root_dir_adl, subdir)
                for subdir2 in os.listdir(subdir_path):
                    subdir2_path = os.path.join(subdir_path, subdir2)
                    
                    data = pd.read_csv(subdir2_path)
                    
                    # There is no need to check, we already know that it is adl and not falls 
                    X_test_adl.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))
                    Y_test_adl.append(0)
                    
        #print("The number of falls in the test dataset is : ", len(Y_test_fall))
        #print("The number of adl in the test dataset is : ", len(Y_test_adl))
        
        # We concatenate to have a single test_data
        X_test = X_test_fall + X_test_adl
        Y_test = Y_test_fall + Y_test_adl
        
        X_test = np.asarray(X_test)
        Y_test = np.asarray(Y_test).reshape(-1, )
        
        enc = LabelEncoder()

        Y_test = enc.fit_transform(Y_test)
        
        test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
        
        os.makedirs("data_division/MobiAct/test_dataset")
                    
        with open('data_division/MobiAct/test_dataset/test_data.pickle', 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

          
        return self.test_clients, len(Y_test_fall), len(Y_test_adl)
      
    def volunteer_based_division(self):
        
        Dict_users = {}
        enc = LabelEncoder()
        
        columns = ['FOL', 'FKL', 'SDL', 'BSC', 'STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO', 'LYI']

        num_sample_per_user = {f'client {i}': {col: [] for col in columns} for i in self.FL_clinets}
        
        for client_num in tqdm(self.FL_clinets) :
            root_dir_fall = "data_division/MobiAct/Users/User"+str(client_num)+"/Fall"
        
            X_train_fall_user = []
            Y_train_fall_user = []
            X_train_adl_user = []
            Y_train_adl_user = []  
            
            for subdir in os.listdir(root_dir_fall):
                subdir_path = os.path.join(root_dir_fall, subdir) # We will receive the types of "Falls" in this step
                nombre = 0
                for subdir2 in os.listdir(subdir_path): 
                    nombre = nombre + 1
                    subdir2_path = os.path.join(subdir_path, subdir2) # We will receive the CSV files
                    data = pd.read_csv(subdir2_path) # Read the time serie
                    cnt = collections.Counter(data['Label'])
                    num_adl = cnt[0]
                    num_fall = cnt[1]

                    if num_fall == 30:
                        X_train_fall_user.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))
                        Y_train_fall_user.append(1)
                    else :
                        X_train_adl_user.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))
                        Y_train_adl_user.append(0)        
            
                num_sample_per_user['client '+str(client_num)][subdir] = nombre
                
            root_dir_adl = "data_division/MobiAct/Users/User"+str(client_num)+"/ADL"

            for subdir in os.listdir(root_dir_adl):
                subdir_path = os.path.join(root_dir_adl, subdir)
                nombre = 0
                for subdir2 in os.listdir(subdir_path):
                    nombre = nombre + 1
                    subdir2_path = os.path.join(subdir_path, subdir2)
                    data = pd.read_csv(subdir2_path)
                    X_train_adl_user.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))
                    Y_train_adl_user.append(0)
                    
                num_sample_per_user['client '+str(client_num)][subdir] = nombre   
                 
            X_train_user = X_train_fall_user + X_train_adl_user
            Y_train_user = Y_train_fall_user + Y_train_adl_user
            
            X_train_user = np.asarray(X_train_user)
            Y_train_user = np.asarray(Y_train_user).reshape(-1, )
            Y_train_user = enc.fit_transform(Y_train_user)
            
            
            Dict_users["User"+str(client_num)] = TensorDataset(torch.from_numpy(X_train_user), torch.from_numpy(Y_train_user))
        
        os.makedirs("data_division/MobiAct/volunteer-based_division")

        # Save the user dict
        with open('data_division/MobiAct/volunteer-based_division/dict_users.pickle', 'wb') as handle:
            pickle.dump(Dict_users, handle, protocol=pickle.HIGHEST_PROTOCOL)
         
        columns_to_keep = ['FOL', 'FKL', 'SDL', 'BSC', 'STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO', 'LYI']
        dict_users_sample = {}
        for client, data in num_sample_per_user.items():
            user_data = [data[column] for column in columns_to_keep]
            dict_users_sample[client] = user_data
        
        for key, value in dict_users_sample.items():
            if isinstance(value, list):  # Check if the value is a list
                for i in range(len(value)):
                    if isinstance(value[i], list) and not value[i]:  # Check if it's an empty list
                        dict_users_sample[key][i] = 0  # Replace empty list with 0
           
        new_dict_users_sample = {'client {}'.format(i): value for i, (key, value) in enumerate(dict_users_sample.items(), start=1)}
         
        new_dict_users_sample['client 10'], new_dict_users_sample['client 12'] = new_dict_users_sample['client 12'], new_dict_users_sample['client 10']
        
        return new_dict_users_sample
    
            
    def unbalanced_division(self, seed, min_samples = 1):
        
        Dict_users = {}
        enc = LabelEncoder()
        random.seed(seed)
        
        for client_num in tqdm(self.FL_clinets) :
            
            root_dir_fall = "data_division/MobiAct/Users/User"+str(client_num)+"/Fall"
        
            X_train_fall_user = []
            Y_train_fall_user = []
            X_train_adl_user = []
            Y_train_adl_user = []  
            
            for subdir in os.listdir(root_dir_fall):
                subdir_path = os.path.join(root_dir_fall, subdir) # We will receive the types of "Falls" in this step
                subfolders2 = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
                
                if (len(subfolders2) != 0):
                    random_subfolders2 = random.sample(subfolders2, k=random.randint(min_samples,len(subfolders2)))
                    for subdir2 in random_subfolders2: 
                        subdir2_path = os.path.join(subdir_path, subdir2) # We will receive the CSV files
                        data = pd.read_csv(subdir2_path) # Read the time serie
                        cnt = collections.Counter(data['Label'])
                        num_adl = cnt[0]
                        num_fall = cnt[1]

                        if num_fall == 30:
                            X_train_fall_user.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))
                            Y_train_fall_user.append(1)
                        else :
                            X_train_adl_user.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))
                            Y_train_adl_user.append(0)        
            
            root_dir_adl = "data_division/MobiAct/Users/User"+str(client_num)+"/ADL"

            for subdir in os.listdir(root_dir_adl):
                subdir_path = os.path.join(root_dir_adl, subdir)
                subfolders2 = files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
                if (len(subfolders2) != 0):
                    random_subfolders2 = random.sample(subfolders2, k=random.randint(min_samples,len(subfolders2)))
                    for subdir2 in random_subfolders2:
                        subdir2_path = os.path.join(subdir_path, subdir2)
                        data = pd.read_csv(subdir2_path)
                        X_train_adl_user.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))
                        Y_train_adl_user.append(0)
                    
            X_train_user = X_train_fall_user + X_train_adl_user
            Y_train_user = Y_train_fall_user + Y_train_adl_user
            
            X_train_user = np.asarray(X_train_user)
            Y_train_user = np.asarray(Y_train_user).reshape(-1, )
            Y_train_user = enc.fit_transform(Y_train_user)
            
            Dict_users["User"+str(client_num)] = TensorDataset(torch.from_numpy(X_train_user), torch.from_numpy(Y_train_user))
            
        os.makedirs("data_division/MobiAct/unbalanced_division")
        
        # Save the user dict
        with open('data_division/MobiAct/unbalanced_division/dict_users.pickle', 'wb') as handle:
            pickle.dump(Dict_users, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def label_skew_division(self, seed, number_types_per_clients):
        
        Dict_users = {}
        enc = LabelEncoder()
        random.seed(seed)
        
        columns = ['FOL', 'FKL', 'SDL', 'BSC', 'STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO', 'LYI']

        num_sample_per_user = {f'client {i}': {col: [] for col in columns} for i in self.FL_clinets}
        
        for client_num in tqdm(self.FL_clinets) :

            root_dir = "data_division/MobiAct/Users/User"+str(client_num)
        
            subdirs = []
            for subdir in os.listdir(root_dir): #Fall and ADL
                subdir_path = os.path.join(root_dir, subdir)
                # check if it is a directory
                for subsubdir in os.listdir(subdir_path): # The types of Fall and ADL
                    subsubdir_path = os.path.join(subdir_path, subsubdir)
                    subfolders2 = [f for f in os.listdir(subsubdir_path) if os.path.isfile(os.path.join(subsubdir_path, f))]
                    if(len(subfolders2) != 0):
                        subdirs.append(subsubdir_path)

            if (len(subdirs) > number_types_per_clients) :
                random_types = random.sample(subdirs, number_types_per_clients)
            else :
                random_types = random.sample(subdirs, len(subdirs))

            #print("client_num : ", client_num, "list ADL : ",random_types)
            my_dict = { 'FOL': 0, 'FKL': 0, 'SDL': 0, 'BSC': 0, 'STD': 0, 'WAL': 0, 'JOG': 0, 'JUM': 0, 'STU': 0, 'STN': 0, 'SCH': 0, 'SIT': 0, 'CHU': 0, 'CSI': 0, 'CSO': 0, 'LYI': 0}
            
            gentypes_name = [os.path.basename(os.path.dirname(subdir_path)) for subdir_path in random_types]
            small_type = [os.path.basename(os.path.split(subdir_path)[1]) for subdir_path in random_types]
                        
            X_train_user = []
            Y_train_user = []

            for subdir_path in random_types:
                cmpt = 0
                parts = subdir_path.split('\\')
                target = parts[-1]
                for subdir2 in os.listdir(subdir_path): 
                    cmpt = cmpt + 1
                    subdir2_path = os.path.join(subdir_path, subdir2)  # We will receive the CSV files
                    data = pd.read_csv(subdir2_path) # Read the time serie
                    
                    X_train_user.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))

                    if (os.path.basename(os.path.dirname(subdir_path)) == "Fall") :
                        
                        cnt = collections.Counter(data['Label'])
                        num_adl = cnt[0]
                        num_fall = cnt[1]

                        if num_fall == 30:
                            Y_train_user.append(1)
                        else :
                            Y_train_user.append(0)  
                    else :
                        Y_train_user.append(0)
            
                num_sample_per_user['client '+str(client_num)][target] = cmpt
               
            X_train_user = np.asarray(X_train_user)
            #print("length totale", len(X_train_user), "length of each : ", taille_donnees)
            Y_train_user = np.asarray(Y_train_user).reshape(-1, )
            Y_train_user = enc.fit_transform(Y_train_user)      
            Dict_users["User"+str(client_num)] = TensorDataset(torch.from_numpy(X_train_user), torch.from_numpy(Y_train_user))            
        
        os.makedirs("data_division/MobiAct/label_skew_division"+str(number_types_per_clients))
        
        # Save the user dict
        with open('data_division/MobiAct/label_skew_division'+str(number_types_per_clients)+'/dict_users.pickle', 'wb') as handle:
            pickle.dump(Dict_users, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        
        columns_to_keep = ['FOL', 'FKL', 'SDL', 'BSC', 'STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO', 'LYI']
        dict_users_sample = {}
        for client, data in num_sample_per_user.items():
            user_data = [data[column] for column in columns_to_keep]
            dict_users_sample[client] = user_data
        
        for key, value in dict_users_sample.items():
            if isinstance(value, list):  # Check if the value is a list
                for i in range(len(value)):
                    if isinstance(value[i], list) and not value[i]:  # Check if it's an empty list
                        dict_users_sample[key][i] = 0  # Replace empty list with 0
           
        new_dict_users_sample = {'client {}'.format(i): value for i, (key, value) in enumerate(dict_users_sample.items(), start=1)}
         
        new_dict_users_sample['client 10'], new_dict_users_sample['client 12'] = new_dict_users_sample['client 12'], new_dict_users_sample['client 10']
        
        return new_dict_users_sample
    
       
    def label_skew_unbalanced_division(self, seed, number_types_per_clients, min_samples = 1):
        
        Dict_users = {}
        enc = LabelEncoder()
        random.seed(seed)
        
        for client_num in tqdm(self.FL_clinets)  :
            root_dir = "data_division/MobiAct/Users/User"+str(client_num)
        
            subdirs = []
            for subdir in os.listdir(root_dir): #Fall and ADL
                subdir_path = os.path.join(root_dir, subdir)
                # check if it is a directory
                for subsubdir in os.listdir(subdir_path): # The types of Fall and ADL
                    subsubdir_path = os.path.join(subdir_path, subsubdir)
                    subfolders2 = [f for f in os.listdir(subsubdir_path) if os.path.isfile(os.path.join(subsubdir_path, f))]
                    if(len(subfolders2) != 0):
                        subdirs.append(subsubdir_path)

            random_types = random.sample(subdirs, min(len(subdirs), number_types_per_clients))
            gentypes_name = [os.path.basename(os.path.dirname(subdir_path)) for subdir_path in random_types]
            small_type = [os.path.basename(os.path.split(subdir_path)[1]) for subdir_path in random_types]
                        
            X_train_user = []
            Y_train_user = []
            

            for subdir_path in random_types:
                subfolders2 = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
                random_subfolders2 = random.sample(subfolders2, k=random.randint(min_samples,len(subfolders2)))
                for subdir2 in random_subfolders2: 
                    subdir2_path = os.path.join(subdir_path, subdir2)  # We will receive the CSV files
                    data = pd.read_csv(subdir2_path) # Read the time serie
                    
                    X_train_user.append(np.array(data[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']]))

                    if (os.path.basename(os.path.dirname(subdir_path)) == "Fall") :
                        
                        cnt = collections.Counter(data['Label'])
                        num_adl = cnt[0]
                        num_fall = cnt[1]

                        if num_fall == 30:
                            Y_train_user.append(1)
                        else :
                            Y_train_user.append(0)  
                    else :
                        Y_train_user.append(0)

            X_train_user = np.asarray(X_train_user)
            Y_train_user = np.asarray(Y_train_user).reshape(-1, )
            Y_train_user = enc.fit_transform(Y_train_user)
                        
            Dict_users["User"+str(client_num)] = TensorDataset(torch.from_numpy(X_train_user), torch.from_numpy(Y_train_user))            
        
        os.makedirs("data_division/MobiAct/label_skew_unbalanced_division")
        
        # Save the user dict
        with open('data_division/MobiAct/label_skew_unbalanced_division/dict_users.pickle', 'wb') as handle:
            pickle.dump(Dict_users, handle, protocol=pickle.HIGHEST_PROTOCOL)
