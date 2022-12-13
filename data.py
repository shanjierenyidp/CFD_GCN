import os
# import pickle
from pathlib import Path
import numpy as np
import torch
from torch._six import container_abcs, string_classes, int_classes
from torch_geometric.data import Data, Batch, Dataset
from mesh_utils import get_mesh_graph
from mesh_utils import cl_dist_graph
import pyvista as pv
import shutil 
from os.path import join as osj
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--root', default='./data/cases',
                        help='Directory of source data.')
    parser.add_argument('--data-dir', '-d', default='./data/dataset0',
                        help='Directory to store graph dataset.')
    parser.add_argument('--fine-mesh-file-name', '-fm', default='./data/aorta3.su2',
                        help='file name of the fine mesh.')
    parser.add_argument('--datasize', '-ts', type=int, default=0)

    args = parser.parse_args()
    return args

def GetData(input_file):
    # print(input_file)
    mesh = pv.read(input_file)
    # print(mesh)
    nodal_features = {}
    nodal_features['Pressure'] = mesh.point_data['Pressure']
    nodal_features['Velocity'] = mesh.point_data['Velocity']
    # nodal_features['Vorticity'] = mesh.point_data['Vorticity']
    return nodal_features

def CreateGraphDataset(root,mesh_file_name, data_dir='./data/dataset0',  start_ts=0, end_ts=1000, inlet_file = './data/inlet_x0.2_y0.5.npz'):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    # load in mesh info
    nodes, edges, elems, marker_dict = get_mesh_graph(mesh_file_name)
    # print(marker_dict.keys())
    # marker_inds = list(set(sum(marker_dict['inlet'], [])))
    rdf, sdf = cl_dist_graph(nodes)

    # load in labels info 
    
    label_collector1 = []
    label_collector2 = []
    # label_collector3 = []
    inlet_info_pack = np.load(inlet_file) # inlet_ensemble,inlet_ensemble_scale, omegaVecxys, scale
    my_encxy = inlet_info_pack['arr_2']
    my_scale = inlet_info_pack['arr_3']

    for i in tqdm(range(start_ts, end_ts)):
        file_name = osj(root,'flow_ts{:d}.vtk'.format(i))
        nodal_features = GetData(file_name)
        label_collector1.append(nodal_features['Pressure'])
        label_collector2.append(nodal_features['Velocity'])
        # add the inlet encoding and scale 
        bcvector = np.append(my_encxy[i,:,:].flatten(),my_scale[i])
        bcvector_feature = np.repeat(bcvector[np.newaxis,:], len(nodes), axis = 0)
        
        # print(bcvector.shape, bcvector_feature.shape)
        # label_collector3.append(nodal_features['Vorticity'])
        # put info in a graph 
        readme = {'nodal_features':'coordx, coordy, rdf, sdf, bcvector',
                # 'edge_features':'d_coordinatex, d_coordinatey, d_coordinatez, d_coordinatenorm',
                'nodal_pos':'corrdx, corrdy',
                'nodal_pressure':'pressure',
                'nodal_velocity':'velocity'
                # 'nodal_vorticity':'vorticity'
                }
        mesh_graph = Data(x = torch.tensor(np.hstack((nodes,rdf[:,np.newaxis],sdf[:,np.newaxis], bcvector_feature)), dtype = torch.float32), 
                          edge_index = torch.tensor(edges, dtype = torch.long), # this should be integer
                          nodal_pressure = torch.tensor(nodal_features['Pressure'],dtype = torch.float32), 
                          nodal_velocity = torch.tensor(nodal_features['Velocity'][:,:2],dtype = torch.float32), 
                        #   nodal_vorticity = torch.tensor(nodal_features['Vorticity'],dtype = torch.float32), 
                          pos = torch.tensor(nodes, dtype = torch.float32))
        torch.save(mesh_graph, osj(data_dir, 'data_{:05d}.pt'.format(i)))
    label_collector1 = np.array(label_collector1)
    label_collector2 = np.array(label_collector2)
    # label_collector3 = np.array(label_collector3)
    min1,max1 = np.min(label_collector1),np.max(label_collector1)
    min2,max2 = np.min(label_collector2.reshape(-1,3)[:,:2],axis=0),np.max(label_collector2.reshape(-1,3)[:,:2],axis=0)
    # min3,max3 = np.min(label_collector3),np.max(label_collector3)
    torch.save({"Pressure": [torch.tensor(min1,dtype=torch.float32),torch.tensor(max1,dtype=torch.float32)],
                "Velocity": [torch.tensor(min2,dtype=torch.float32),torch.tensor(max2,dtype=torch.float32)]
                # "Vorticity": [torch.tensor(min3,dtype=torch.float32),torch.tensor(max3,dtype=torch.float32)]
                }, 
                osj(data_dir, 'data_range.pt'))
    
    return mesh_graph, readme

if __name__ == '__main__':
    print('hi, I am recreating the data')
    args = parse_args()
    CreateGraphDataset(args.root,args.fine_mesh_file_name, data_dir=args.data_dir,  end_ts=args.datasize)
    
# Run it in cfd-gcn
# python data.py --root './data/cases' -d './data/dataset0' -fm './data/aorta3.su2' -ts 2000