# running in python env: cfd-gcn 
# please run su2wpcmdf alias in advance 
import math
import sys
from typing import Tuple

import os
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import SU2
from mpi4py import MPI  # Must be imported before pysu2 or else MPI error happens at some point
import pysu2
import pysu2ad
from mesh_utils import get_mesh_graph


print('Hello')

class myClass():
    def __init__(self, config_file,mesh_file):
        self.forward_config = config_file
        self.mesh_file = mesh_file
        self.su2 = pysu2.CSinglezoneDriver(self.forward_config, 1, 2, MPI.COMM_SELF)
    def forward(self, x):
        return 2*x

class myClassad():
    def __init__(self,config_file_ad, mesh_file):
        self.adjoint_config = config_file_ad
        self.mesh_file = mesh_file
        self.su2ad= pysu2ad.CDiscAdjSinglezoneDriver(self.adjoint_config, 1, 2, MPI.COMM_SELF)
    def forward(self, x):
        return 2*x
# forward_driver = pysu2.CSinglezoneDriver(worker_forward_config, num_zones, dims, MPI.COMM_SELF)  





if __name__ == '__main__':
    
    
    # calculate direct 
    # config_file  = './coarse.cfg'
    config_file  = './aorta_2d_coarse_little.cfg'
    config_file_ad  = './aorta_2d_coarse_ad_little.cfg'
    mesh_file = './data/aorta12_flip.su2'
    mesh_graph = get_mesh_graph(mesh_file)
    mysu2 = myClass(config_file,mesh_file)
    nodesx = torch.from_numpy(mesh_graph[0][:,0])
    
    
    ###
    nodesx = torch.tensor([0,0,0])
    ###
    
    
    nodesy = torch.from_numpy(mesh_graph[0][:,1])
    nodesy = torch.tensor([0,0,0])
    temp = torch.tensor(0.2)
    print('python : here is the input x and y for the coarse mesh',nodesx.shape)
    print('python : here is the number of inputs', mysu2.su2.GetnDiff_Inputs())
    print('python : here is the number of outputs', mysu2.su2.GetnDiff_Outputs())
    # setting the value of the diffinputs 
    mysu2.su2.SetDiff_Inputs_Vars(nodesx.flatten().tolist(),0)
    mysu2.su2.SetDiff_Inputs_Vars(nodesy.flatten().tolist(),1)
    mysu2.su2.SetDiff_Inputs_Vars([temp.tolist()],2)

    # register variable
    print('python : registering variable')
    mysu2.su2.ApplyDiff_Inputs_Vars()
    # run simulation 
    print('python : running simulation')
    mysu2.su2.StartSolver()
    
    # get output
    print('get output')
    output0 = np.array(mysu2.su2.GetDiff_Outputs_Vars(0))
    output1 = np.array(mysu2.su2.GetDiff_Outputs_Vars(1))
    output2 = np.array(mysu2.su2.GetDiff_Outputs_Vars(2))
    # output3 = np.array(mysu2.su2.GetDiff_Outputs_Vars(3))
    
    print('here is the direct run output')
    print(output0.shape)
    print(output1.shape)
    print(output2.shape)
    # print(output3.shape)
    
    #rename restart of direct flow to solution flow 
    # os.system("mv restart_flow_00000.dat solution_flow_00000.dat")
    os.system("mv ./little/restart_flow.dat ./little/solution_flow.dat")
    
    
    
    # sys.exit('stop')
    
    mysu2 = myClassad(config_file_ad,mesh_file)
    
    #adjoint:
    mysu2.su2ad.SetDiff_Inputs_Vars(nodesx.flatten().tolist(),0)
    mysu2.su2ad.SetDiff_Inputs_Vars(nodesy.flatten().tolist(),1)
    mysu2.su2ad.SetDiff_Inputs_Vars([temp.tolist()],2)
    
    # register variable
    print('python : registering variable ad')
    mysu2.su2ad.ApplyDiff_Inputs_Vars()
    # run simulation 
    print('python : running simulation ad')
    mysu2.su2ad.StartSolver()
    
    # get output
    print('get output')
    outputad0 = np.array(mysu2.su2ad.GetDiff_Outputs_Vars(0))
    outputad1 = np.array(mysu2.su2ad.GetDiff_Outputs_Vars(1))
    outputad2 = np.array(mysu2.su2ad.GetDiff_Outputs_Vars(2))
    
    
    print('here is the adjoint run output')
    print(outputad0.shape)
    print(outputad1.shape)
    print(outputad2.shape) 
    # print(outputad1) 
    
    print(outputad0)
    
    # print('here are the specific outputs')
    # # print(output0)
    # # print(outputad0)
    # # print(outputad0-output0)
    # print('number of diff inputs:', mysu2.su2ad.GetnDiff_Inputs())
    # print('diff inputs for vetex 2 is', mysu2.su2ad.GetTotal_Sens_Diff_Inputs(1))
    # print('diff inputs for vetex 2s shape is', np.array(mysu2.su2ad.GetTotal_Sens_Diff_Inputs(1)).shape)

    # print('diff inputs for vetex 3 is', mysu2.su2ad.GetTotal_Sens_Diff_Inputs(2))
    # print('diff inputs for vetex 3s shape is', np.array(mysu2.su2ad.GetTotal_Sens_Diff_Inputs(2)).shape)






# python runlittlecase.py 
