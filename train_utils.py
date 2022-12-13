import torch
import numpy as np
import pyvista as pv

# only normalizing the pressure/labels here
# class Normalize_dataset():
#     def __init__(self,data, input_method = None, output_method = None):
#         self.data = data
#         self.mean = torch.mean(self.data)
#         self.std =  torch.std(self.data)
#     def pre_normalize(self):
#         self.norm_data = (self.data - self.mean)/self.std
        
#     def normalize(self, new_data):
#         norm_new_data = (new_data - self.mean)/self.std
#         return norm_new_data
#     def denormalize(self, new_data):
#         denorm_new_data = new_data*self.std + self.mean
#         return denorm_new_data

# training
def train(model, device, train_loader, optimizer, criterion, scheduler = None, freeze_mesh = False, is_cw = None, is_nan =None, note = 0):
    # define trainloader
    
    model.train()
    train_ins_error = []
    nan_ids_list = []
    flip_ids_list = []
    for batch_idx, batch in enumerate(train_loader):
        batch = transfer_batch_to_device(batch, device)
        optimizer.zero_grad()
        # for k,v in enumerate(batch):
        #     print(k,v)
        #     print(v[-1])
        # for k,v in enumerate(model.pre_convs):
        #     print(v.parameters())
        # for k,v in enumerate(model.convs):
        #     print(v.parameters())
            
            
            
        output = model(batch)
        labels = batch.y
        
        print(labels[0].size)
        
        print('print model mes inputs output size:' , output.size())
        print('print model mes inputs label size:', labels.size())
        
        loss = criterion(output, labels)
        loss.backward()
        ### optimize ###
        ## freeze boudnary ##
        if freeze_mesh == False:
            if model.nodes.grad is not None:
                # do not optimize airfoil nodes to maintain shape
                model.nodes.grad[model.marker_inds] = 0
        else:
            model.nodes.grad = 0
        # save prev nodes for mesh checking below
        prev_nodes = model.nodes.detach().clone()
        optimizer.step()
        nodes = model.nodes
        # nodes_list.append(nodes)
        elems = model.elems[0]
        print('at epoch',note,'print pre and post nodes', prev_nodes, nodes)
        
        nan_ids = is_nan(nodes) ; print('nan_ids', nan_ids, 'at epoch', note);nan_ids_list.append(nan_ids)
        with torch.no_grad():
            nodes[nan_ids] = prev_nodes[nan_ids]
        flipped_elems = is_cw(nodes, elems).nonzero()
        while flipped_elems.shape[0] > 0:
            flipped_inds = [elems[i] for i in flipped_elems]
            flipped_inds = torch.tensor(flipped_inds).unique(); print('flipped_inds',flipped_inds) ;flip_ids_list.append(flipped_inds)
            with torch.no_grad():
                nodes[flipped_inds] = prev_nodes[flipped_inds]
            flipped_elems = is_cw(nodes, elems).nonzero()        
        # nodes_fix_list.append(nodes)
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
        return torch.mean(torch.stack(train_ins_error)), nan_ids_list, flip_ids_list


def test(model,device,loader, my_normalizer, data_maxmin):
    # model.to('cpu')
    test_errors = []
    # prediction_denorms = []
    for batch_idx, batch in enumerate(loader):
        
        batch = transfer_batch_to_device(batch, device)
        prediction= model(batch).detach().cpu()
        labels = batch.y.detach().cpu()
        
        print('prediction shape', prediction.size())  # 926,3
        prediction_denorm = my_normalizer(prediction,data_maxmin)
        # prediction_denorms.append(prediction_denorm)
        labels_denorm = my_normalizer(labels,data_maxmin)
        
        test_error  = np.linalg.norm((prediction_denorm.detach().cpu().numpy()-labels_denorm.detach().cpu().numpy()))/ np.linalg.norm(labels_denorm.detach().cpu().numpy())
        test_errors.append(test_error**2)
    test_errors_mean  = np.mean(test_errors)
    # prediction_denorms = torch.stack(prediction_denorms).view(-1,prediction_denorm.size()[-1])
    return test_errors, test_errors_mean

from torch import nn, optim

def configure_optimizers(model, optm_name, lr = 0.01):
    if  optm_name == 'adam':
        optimizers = [optim.Adam(model.parameters(), lr=lr)]
    if  optm_name == 'rmsprop':
        optimizers = [optim.RMSprop(model.parameters(), lr=lr)]
    if  optm_name == 'sgd':
        optimizers = [optim.SGD(model.parameters(), lr=lr)]
    schedulers = []
    return optimizers, schedulers



def transfer_batch_to_device(batch, device):
    for k, v in batch:
        # print(k,v)
        if hasattr(v, 'to'):
            # print('push to gpu')
            batch[k] = v.to(device)
    return batch

# import sys
# sys.path.insert(0, '/home/pandu/Panresearch_local/PPP_Utility')
# import panpv as pm

def node2mesh(nodes, output_filename = 'outputmesh.vtk', template_mesh_filename = './data/aorta3.stl', write = False):
    template_mesh = pv.read(template_mesh_filename)
    # print(nodes.size())
    mesh_nodes = np.hstack((nodes.detach().cpu().numpy(),np.repeat(0.,nodes.size()[0])[:,np.newaxis]))
    template_mesh.points = mesh_nodes

    if write == True:
        template_mesh.save(output_filename)
    return template_mesh


# def data2mesh(prediction,label, output_filename, template_mesh_filename = './data/aorta3.stl'):
def data2mesh(datas,tags, output_filename = 'outputmesh.vtk', template_mesh_filename = './data/aorta3.stl', write = False):
    template_mesh = pv.read(template_mesh_filename)
    assert len(datas) == len(tags), "point data and tags must have the same dimension"
    for i in range(len(tags)):
        template_mesh.point_data[tags[i]] = datas[i]
    if write == True:
        template_mesh.save(output_filename)
    return template_mesh
    
def visualize_vtk(mesh, tag, scale =1,window_size = [800,800]):
    tag_data = mesh.point_data[tag]
    assert tag_data != [], 'make sure tag is included in mesh'
    #plot using pyvista
    pl = pv.Plotter(window_size = window_size)
    # pl.background_color = 'w'
    #pl.add_points(point, color = 'red',point_size = 5)
    
    if len(tag_data.shape)==2 and tag_data.shape[-1] == 3:
        # glyphs = mesh.glyph(orient='vectors', scale='scalars', factor=0.003, geom=pv.Arrow())
        mesh.set_active_vectors(tag)
        mesh.point_data['tagscale'] = tag_data* scale
        mesh.set_active_vectors('tagscale')
        print(mesh.arrows.point_data)
        pl.add_mesh(mesh.arrows, lighting=False, scalar_bar_args={'title': "Vector Magnitude"})
        pl.add_mesh(mesh, color = 'gray',show_edges = True, opacity=0.3)
        
    elif len(tag_data.shape)==2 and tag_data.shape[-1] == 2:
        tag_data = np.hstack((tag_data, np.repeat(0,len(tag_data))[:,np.newaxis]))
        mesh.point_data['tagscale'] = tag_data* scale
        mesh.set_active_vectors('tagscale')
        print(mesh.arrows.point_data)
        pl.add_mesh(mesh.arrows, lighting=False, scalar_bar_args={'title': "Vector Magnitude"})
        pl.add_mesh(mesh, color = 'gray',show_edges = True, opacity=0.3)
        
    elif len(tag_data.shape)==1:
        pl.add_mesh(mesh,color = 'gray',show_edges = True, opacity=1, scalars = tag_data)# scalars = ?
        
    pl.enable_anti_aliasing()
    pl.show(jupyter_backend='panel')

