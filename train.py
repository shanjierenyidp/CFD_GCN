import os
import logging
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.unpool import knn_interpolate

# from su2torch import SU2Module
from mesh_utils import write_graph_mesh, quad2tri, get_mesh_graph, signed_dist_graph,is_nan, is_cw
from os.path import join as osj
from tqdm import tqdm
from train_utils import *
from models import * 
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.data import DataLoader
# from su2torch import activate_su2_mpi
from su2_function_mdf import SU2Module
from su2_function_mpi_mdf import activate_su2_mpi
from dataset import AortaDataset
from train_utils import *
import shutil

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
print('starting activate_su2_mpi')
activate_su2_mpi(remove_temp_files=True)
print('activate_su2_mpi finished')

## create dataset 
my_dataset = AortaDataset('./data/dataset0','All')

# parameters to creat CFDGCN model 
su2_config = 'aorta_2d_coarse.cfg'
coarse_mesh = './data/aorta12_flip.su2'
hidden_channels = 512
num_layers = 6 #
initial_channels = 11
num_end_convs = 3
out_channels = 3 # 1 for pressure, 2 for velocity, 3 for all
freeze_mesh = False
gpus=1
device='cuda' if gpus > 0 else 'cpu'
model = CFDGCN(su2_config,
               coarse_mesh,
               hidden_channels=hidden_channels,
               initial_channels = initial_channels,
               num_convs=num_layers,
               num_end_convs=num_end_convs,
               out_channels=out_channels,
               freeze_mesh=freeze_mesh,
               device=device)
model.to(device)

# check model parametes
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data.size())


expi = 1 # experiment id
snum=2000 # total number of samples
# snum=500
epochs = 3000 # training epoches 
data_size = snum # data_size is the same as snum
batch_size = 4 # the batch size for the training 
print(len(my_dataset)) # check batch size 

# create datasets 
validation_split_ratio = 0.9; test_split_ratio= 0.99;shuffle_dataset = True; random_seed = 0
dataset_size = len(my_dataset)
indices = list(range(dataset_size))
validation_split = int(np.floor(validation_split_ratio * dataset_size))
test_split = int(np.floor(test_split_ratio * dataset_size))

# use the follow code if you want to shuffle the dataset
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
train_indices, validation_indices, test_indices = indices[:validation_split], indices[validation_split:test_split], indices[test_split:]

train_dataset = torch.utils.data.Subset(my_dataset, train_indices)
validation_dataset = torch.utils.data.Subset(my_dataset, validation_indices)
test_dataset = torch.utils.data.Subset(my_dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=1)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=1)


### optm ###
lr =0.01  # initial learning rate

optimizer, _= configure_optimizers(model, 'adam', lr = lr)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer[0],milestones=np.linspace(0,3000,120,dtype = int), gamma=0.926)
output_dir = './results/test_results_{:d}_{:d}_{:d}_{:d}'.format(expi, snum, epochs, batch_size)
# ################################################################################## 
# creating the experiment folder
print('creating the test folder')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir+'/show')
os.mkdir(output_dir+'/prediction')

### end creating 


train_errors = []
temp_meshes = []
for i in tqdm(range(epochs)):
    # print('yo here is the temp mesh coords',model.nodes)
    
    temp_err, nan_ids_list, flip_ids_list = train(model, device, train_loader, optimizer[0], criterion, scheduler = scheduler, freeze_mesh = False, is_cw = is_cw,is_nan = is_nan, note = i)
    temp_mesh = node2mesh(model.nodes, output_filename =output_dir+'/show/show{:d}.vtk'.format(i), template_mesh_filename = './data/aorta12_flip.vtp', write = True)
    # print('yo here is the temp mesh',temp_mesh)
    # model.nodes
    # shutil.move(output_dir+'/train/show.vtk',output_dir+'/train/show{:d}.vtk'.format(i))
    train_errors.append(temp_err)
    temp_meshes.append(temp_mesh)
    
print('mesh change is', np.max(abs(temp_meshes[-1].points - temp_meshes[0].points)))
train_errors_tensor = torch.stack(train_errors)
train_errors_np = train_errors_tensor.detach().cpu().numpy()


#sec.save the model 
torch.save({
            'epoch': len(train_errors),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer[0].state_dict(),
            'loss': train_errors,
            }, output_dir+"/checkpoint.pt")

# end
#########################################################################################
#sec.save the training model results 
checkpoint = torch.load(output_dir+"/checkpoint.pt")
train_errors_np =torch.stack(checkpoint['loss']).detach().cpu().numpy()

#printing the training error curve 
#figure single 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
fig, ax = plt.subplots(figsize=(10,8))
lw=3
ax.plot(train_errors_np ,color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='')
ax.set_yscale('log')
ax.set_xlabel('log scale error')
ax.set_ylabel('epochs')
fig.savefig(output_dir+'/train_error_curve.png',bbox_inches='tight')

# calculate the testing error on the test dataset 
test_errors, test_errors_mean = test(model,device,test_loader, my_dataset.ydenormalize, my_dataset.get_data_maxmin("All"))
print('the averaged mse test error is: ', test_errors_mean**2)

output_test_folder_name = output_dir+'/prediction'
# os.mkdir(output_test_folder_name)
print(output_test_folder_name)
template_mesh = pv.read('./data/cases/flow_ts0.vtk')
model.to('cpu')
for i, batch in enumerate(test_loader):
# for i, batch in enumerate(train_loader):
    label = batch.y
    prediction  = model(batch)
    label_denorm = my_dataset.ydenormalize(label,my_dataset.get_data_maxmin('All'))
    prediction_denorm = my_dataset.ydenormalize(prediction,my_dataset.get_data_maxmin('All'))        
    # print('print comparison',label, prediction )
    # print('print comparison',label_denorm, prediction_denorm )
    template_mesh.point_data['p_U'] = prediction_denorm[:,:2].detach().numpy()
    template_mesh.point_data['l_U'] = label_denorm[:,:2].detach().numpy()
    template_mesh.point_data['d_U'] = prediction_denorm[:,:2].detach().numpy()-label_denorm[:,:2].detach().numpy()
    template_mesh.point_data['p_Um'] = np.linalg.norm(template_mesh.point_data['p_U'],axis = -1)
    template_mesh.point_data['l_Um'] = np.linalg.norm(template_mesh.point_data['l_U'],axis = -1)
    template_mesh.point_data['d_Um'] = np.linalg.norm(template_mesh.point_data['d_U'],axis = -1)
    template_mesh.point_data['p_P'] = prediction_denorm[:,-1].detach().numpy()
    template_mesh.point_data['l_P'] = label_denorm[:,-1].detach().numpy()
    template_mesh.point_data['d_P'] = prediction_denorm[:,-1].detach().numpy()-label_denorm[:,-1].detach().numpy()
    template_mesh.save(output_test_folder_name+'/test{:d}.vtk'.format(i))

##################################################################################
################## to run this code run the following in the terminal#############
##################################################################################
# conda activate cfd-gcn
# su2wpcmdf0
# mpirun -np 5 python train.py > log
# mpirun -np 17 python train.py > log
##################################################################################
