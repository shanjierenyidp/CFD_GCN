import os
import numpy as np
import torch
import re
from torch_geometric.data import Data, Batch, Dataset
from os.path import join as osj
# import pickle

class AortaDataset(Dataset):
    def __init__(self, root, target):
        self.root = root
        self.file_list = os.listdir(self.root)
        self.file_list = sorted(self.file_list)
        self.len = len(self.file_list)-1 # -1 cuz datarange.pt
        self.target = target
        self.normalizer_parameter = torch.load(osj(self.root,self.file_list[-1]))
        self.data_max_P, self.data_min_P = self.normalizer_parameter['Pressure']
        self.data_max_V, self.data_min_V = self.normalizer_parameter['Velocity']
        self.data_max_VP, self.data_min_VP = torch.cat((self.data_max_V,self.data_max_P.unsqueeze(0))), torch.cat((self.data_min_V,self.data_min_P.unsqueeze(0)))
        self.access_mode =  'raw'
        super().__init__(root)

    def __len__(self):
        return self.len

    def set_access_mode(self,input_mode):
        assert input_mode in ["raw","norm"], "input mode must be one of those modes: raw, norm"
        self.access_mode =  input_mode

    def get(self, idx):
        my_idx = int(re.findall(r'\d+', self.file_list[idx])[0])
        data_i = torch.load(osj(self.root,self.file_list[idx]))

        if self.target == 'Pressure':
            if self.access_mode == 'raw':
                data_i.y = data_i.nodal_pressure
            elif self.access_mode == 'norm':    
                data_i.y = self.preprocess(data_i.nodal_pressure)
            data = Data(x = data_i.x,
                        edge_index = data_i.edge_index,
                        y_denorm = data_i.nodal_pressure, 
                        # pressure = data_i.nodal_pressure,
                        # velocity = data_i.nodal_velocity,
                        ts = torch.tensor(my_idx),
                        y=data_i.y)
        elif self.target == 'Velocity':
            if self.access_mode == 'raw':
                data_i.y = data_i.nodal_velocity
            elif self.access_mode == 'norm':    
                data_i.y = self.preprocess(data_i.nodal_velocity)
            # data_i.y = self.preprocess(data_i.nodal_velocity, self.target)
            # data_ip1.y = self.preprocess(data_ip1.nodal_velocity)
            
            # print('print inside dataset.py:')
            # print(data_i.x.dtype)
            # print(data_i.y.dtype)
            # print(data_i.edge_index)
            # print(torch.tensor(100, dtype = torch.float32).dtype)
            # print(torch.tensor(my_idx).dtype)
            # print(data_ip1.y.dtype)
            
            
            data = Data(x = data_i.x, 
                        edge_index = data_i.edge_index,
                        y_denorm = data_i.nodal_velocity, 
                        # pressure = data_i.nodal_pressure,
                        # velocity = data_i.nodal_velocity,
                        ts = torch.tensor(my_idx),
                        y= data_i.y)
        elif self.target == 'All':
            my_all = torch.cat((data_i.nodal_velocity,data_i.nodal_pressure[ :,np.newaxis]), dim = 1)
            if self.access_mode == 'raw':
                data_i.y = my_all
            elif self.access_mode == 'norm':    
                data_i.y = self.preprocess(my_all)
            # my_all = torch.cat((data_i.nodal_velocity,data_i.nodal_pressure[ :,np.newaxis]), dim = 1)
            # data_i.y = self.preprocess(my_all, self.target)
            # data_i.y0 = self.preprocess(data_i.nodal_pressure, 'Pressure')
            # data_i.y1 = self.preprocess(data_i.nodal_velocity, 'Velocity')
            # data_i.y = torch.cat((data_i.y1,data_i.y0[ :,np.newaxis]), dim = 1)
            
            data = Data(x = data_i.x, 
                        edge_index = data_i.edge_index,
                        # pressure = data_i.nodal_pressure,
                        # velocity = data_i.nodal_velocity,
                        y_denorm = my_all,
                        ts = torch.tensor(my_idx),
                        y= data_i.y)
        #     data_i.y = self.preprocess(data_i.nodal_vorticity)
        #     data_ip1.y = self.preprocess(data_ip1.nodal_vorticity)
        #     data = Data(x = torch.cat((data_i.x,data_i.y.unsqueeze(0)), dim=1), 
        #                     edge_index = data_i.edge_index,
        #                     y=data_ip1.y)
        
        
        return data

    def preprocess(self, tensors):
        if self.target == "Pressure":
            data_max, data_min = self.data_max_P, self.data_min_P
        elif self.target == "Velocity":
            data_max, data_min = self.data_max_V, self.data_min_V
        elif self.target == "All":
            data_max, data_min = self.data_max_VP, self.data_min_VP
        return self.ynormalize(tensors,[data_max, data_min])


    def get_data_maxmin(self, tag):
        if tag == "Pressure":
            return [self.data_max_P, self.data_min_P]
        elif tag == "Velocity":
            return [self.data_max_V, self.data_min_V]
        elif tag == "All":
            return [self.data_max_VP, self.data_min_VP]


    @staticmethod
    def ynormalize(new_data, data_maxmin):
        data_max, data_min = data_maxmin[0], data_maxmin[1]
        normalized = (new_data - torch.tensor(data_min)) / (torch.tensor(data_max) - torch.tensor(data_min)) * 2 - 1
        return normalized
    @staticmethod
    def ydenormalize(new_data, data_maxmin):
        data_max, data_min = data_maxmin[0], data_maxmin[1]
        denormalized = (new_data+1)*(torch.tensor(data_max) - torch.tensor(data_min))/2 + torch.tensor(data_min)
        return denormalized

    # def my_postprocess(self, tensors, my_target):
        
    #     data_max, data_min = self.normalization[my_target]
    #     denormalized = (tensors+1)*(torch.tensor(data_max) - torch.tensor(data_min))/2 + torch.tensor(data_min)
        
    #     return denormalized
    
    # def my_denormalize(self,tensors):
        
    #     denormalized_v = self.my_postprocess(tensors[:, :2],'Velocity')
    #     denormalized_p = self.my_postprocess(tensors[:, 2:],'Pressure')
        
    #     return torch.cat((denormalized_v, denormalized_p),dim=1)

    def _download(self):
        pass

    def _process(self):
        pass

if __name__ == '__main__':

    # mydata = AortaDataset('./data/dataset0','Pressure')
    # mydata = AortaDataset('./data/dataset0','Velocity')
    # mydata = AortaDataset('./data/dataset0','All')
    f0 = mydata[0]
    print(f0)
    mydata.set_access_mode("norm")
    f0n = mydata[0]
    print(f0n)
    
    rec_error = f0.y-mydata.ydenormalize(f0n.y)
    print(torch.max(rec_error))
    # print(mydata[1].x.size())
    # print(mydata[1].y.size())
    # print(mydata[1].ts.size())
    
    
    
    