import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

class mobiAct_preprocessing:
    
    def __init__(self):
        pass
    
    def under_sampling_ADL(self, data):
        
        length = len(data) // 4
        
        result = pd.DataFrame(signal.resample(data['acc_x'][:length], 50))
        result.columns = ['acc_x']
        
        result['acc_y'] = pd.DataFrame(signal.resample(data['acc_y'][:length], 50))
        
        result['acc_z'] = pd.DataFrame(signal.resample(data['acc_z'][:length], 50))
        
        result.index = list(range(0, 50))
        
        result['label'] = 0 
        
        return result
    
    def under_sampling_fall(self, data):
        
        std_length = len(data[data['label']=='STD'])
        fall_length = len(data[data['label']=="FOL"])

        fall_data = data[std_length : std_length + fall_length]
        df = fall_data[['acc_x', 'acc_y', 'acc_z']]

        n = 10
        resampling_size = 30

        # For few cases where the length of STD is shorter than 10
        if std_length >= n:
            # 10 samples from STD + FALL resampled into 30 samples + 10 samples from LYI
            # acc_x
            std_x = pd.DataFrame(data['acc_x'][std_length - n : std_length])
            fall_x = pd.DataFrame(signal.resample(df['acc_x'], resampling_size))
            fall_x.columns = ['acc_x']
            lyi_x = pd.DataFrame(data['acc_x'][std_length + fall_length : std_length + fall_length + n])
            result = pd.concat([std_x, fall_x, lyi_x], axis=0)
            
            # acc_y
            std_y = pd.DataFrame(data['acc_y'][std_length - n : std_length])
            fall_y = pd.DataFrame(signal.resample(df['acc_y'], resampling_size))
            fall_y.columns = ['acc_y']
            lyi_y = pd.DataFrame(data['acc_y'][std_length + fall_length : std_length + fall_length + n])
            result['acc_y'] = pd.concat([std_y, fall_y, lyi_y])
            
            # acc_z
            std_z = pd.DataFrame(data['acc_z'][std_length - n : std_length])
            fall_z = pd.DataFrame(signal.resample(df['acc_z'], resampling_size))
            fall_z.columns = ['acc_z']
            lyi_z = pd.DataFrame(data['acc_z'][std_length + fall_length : std_length + fall_length + n])
            result['acc_z'] = pd.concat([std_z, fall_z, lyi_z])
            
            # newly indexing
            result.index = list(range(0, len(result)))

            result.loc[0:10, 'label'] = "STD"
            result.loc[10:40, 'label'] = "FOL"   #initialzing for label
            result.loc[40:50, 'label'] = "LYI"
            #result.loc[n: n + resampling_size - 1, 'label'] = 1    # labelling as 'FALL' = 1

        # If the length of STD is shorter than 10
        else:
            # STD + FALL resampled into 30 samples + 10 samples from LYI
            # acc_x
            std_x = pd.DataFrame(data['acc_x'][: std_length])
            fall_x = pd.DataFrame(signal.resample(df['acc_x'], resampling_size))
            fall_x.columns = ['acc_x']
            lyi_x = pd.DataFrame(data['acc_x'][fall_length : fall_length + n + (n - std_length)])
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
            
            result['label'] = 0   #initialzing for label
            result.loc[std_length:std_length + resampling_size - 1, 'label'] = 1    # labelling as 'FALL' = 1
            
        return result
    
    
    def sliding_window_fall(self, data, n) :
        result = []
        
        for i in range(0, len(data) - n):
    
            temp = list()

            for x, y, z, label in zip(data['acc_x'][i : i + n], data['acc_y'][i : i + n], data['acc_z'][i : i + n], data['label'][i: i + n]):
                temp.append([x]+[y]+[z]+[label])
            
            example_df = pd.DataFrame(temp) # Transform it to pd in order to be saved as csv file
            example_df.columns = ["Accelerometer_x", "Accelerometer_y", "Accelerometer_z", "Label"] # rename the columns
            result.append(example_df)
        
        return result