import matplotlib.pyplot as plt
import pandas as pd

class result_plot:
    
    def __init__(self):
        pass
    
    def assendante_list(self, unelist):
        max = 0
        cpt = 0
        max_list = []
        comm_rond = []
        for i in range(0, len(unelist)):
            if(unelist[i] > max):
                max = unelist[i] 
                max_list.append(unelist[i])
                comm_rond.append(cpt)
            cpt = cpt + 1
        
        if(comm_rond[-1] != len(unelist)):
            max_list.append(max_list[-1])
            comm_rond.append(len(unelist))
            
        return max_list, comm_rond
    
    def cummulative_list(self, unelist):
        cum_list = [0]*len(unelist)
        cum_list[0] = unelist[0]

        for i in range(1,  len(unelist)):
            cum_list[i] = unelist[i] + cum_list[i -1]
            
        return cum_list

        
    