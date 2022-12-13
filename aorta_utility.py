
##sec.import
import pyvista as pv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import panel as pn
pn.extension(comms='vscode')
import vtk
from vtk.util import numpy_support as npvtk
sys.path.insert(0, '/home/pandu/Panresearch_local/PPP_Utility')
from panpv import panmesh as pm
from panpv import pansim as ps


import scipy.stats
from scipy import interpolate
import pygalmesh
import shutil
import json
import subprocess
##end
##sec.defs
def aorta_pts(x,cl,normal,res_r,res_x,probe_resr,probe_resx,probe_cut):
    sk=cl.shape[0]
    radius = np.array([x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[7],x[7],x[7]])
    uw =  cl+normal*np.repeat(radius,3).reshape(sk,3)
    lw =  cl-normal*np.repeat(radius,3).reshape(sk,3)
    lw_flip = np.flip(lw,0)
    inlet = np.concatenate((lw[0][np.newaxis,:],uw[0][np.newaxis,:]),axis=0)
    outlet = np.concatenate((uw[-1][np.newaxis,:],lw[-1][np.newaxis,:]),axis=0)

    uw_spline,lw_spline,lw_spline_flip = np.zeros((res_x,3)),np.zeros((res_x,3)),np.zeros((res_x,3))
    inlet_spline,outlet_spline = np.zeros((res_r,3)),np.zeros((res_r,3))
    outline = np.zeros((2*(res_x+res_r)-4,3))

    uw_spline = pv.Spline(uw,res_x).points
    lw_spline = pv.Spline(lw,res_x).points
    uw_cut = np.argmin(np.linalg.norm(uw_spline-uw[probe_cut],axis=1))
    lw_cut = np.argmin(np.linalg.norm(lw_spline-uw[probe_cut],axis=1))
    # print(cut)

    usl = uw_spline[uw_cut:]
    lsl = lw_spline[lw_cut:]
    def interpolate_line(oldline, N):
        sx,sy,sz = oldline[:,0], oldline[:,1],oldline[:,2]
        arc =np.cumsum(np.insert(np.sqrt((sx[1:]-sx[:-1])**2+(sy[1:]-sy[:-1])**2+(sz[1:]-sz[:-1])**2),0,0))
        fx = interpolate.interp1d(arc, sx)
        fy = interpolate.interp1d(arc, sy)
        fz = interpolate.interp1d(arc, sz)
        arc_new = np.linspace(arc[0], arc[-1], N)
        x_new = fx(arc_new)
        y_new = fy(arc_new)
        z_new = fz(arc_new)
        newline = np.column_stack((x_new,y_new,z_new))
        return newline

    usl_probe= interpolate_line(usl, probe_resx)
    lsl_probe= interpolate_line(lsl, probe_resx)
    dr_probe = (usl_probe-lsl_probe)/(probe_resr-1)
    #construct probe point cloud
    probe_ptc = np.zeros((probe_resr-2,probe_resx,3))
    for i in range(probe_resr-2):
        probe_ptc[i]= (lsl_probe+dr_probe*(i+1))


    lw_spline_flip = np.flip(lw_spline,axis=0)
    inlet_spline = np.linspace(inlet[0,:],inlet[1,:],res_r)
    outlet_spline = np.linspace(outlet[0,:],outlet[1,:],res_r)
    outline=np.concatenate((inlet_spline[0:-1,:],uw_spline[0:-1,:],outlet_spline[0:-1,:],lw_spline_flip[0:-1,:]),axis = 0)
    
    #meshings
    mesh_pts = np.zeros((res_r,res_x,3))
    mesh_pts[0] =lw_spline
    for i in range(res_r-2):
        mesh_pts[i+1] = lw_spline+(uw_spline-lw_spline)*(i+1)/(res_r-1)
    mesh_pts[-1]=uw_spline
    
    aorta = {'inlet':inlet,'outlet':outlet,'uw':uw,'lw':lw_flip,'inlet_spline':inlet_spline,
             'outlet_spline':outlet_spline,'uw_spline':uw_spline,'lw_spline':lw_spline_flip, 
             'outline': outline, 'mesh':mesh_pts, 'probe_ptc': probe_ptc}
    return aorta

def triangulate(aorta,mesh_size,num_lloyd_steps=10, write=False, save_dir=None, file_name = None, version =42, write_vtp=True):
    #randomseed
    # np.random.seed(0)
    # read res from aorta
    resr,resx = len(aorta['inlet_spline']),len(aorta['uw_spline'])
    # use pygalmesh to triangulate
    points_outline = aorta['outline']
    constraints = np.column_stack([np.arange(0,len(points_outline)),np.arange(1,len(points_outline)+1)])%len(points_outline)
    mesh = pygalmesh.generate_2d(points_outline[:,:2],constraints,max_edge_size=mesh_size,num_lloyd_steps=10)
    # pass the tri surface to pyvista
    pv_mesh = pv.wrap(mesh)
    # get edge ids
    # Add points in to mesh_vtk
    points=pv_mesh.points
    points_vtk = vtk.vtkPoints()
    mesh_vtk = vtk.vtkUnstructuredGrid()
    for i in range(len(points)):
        points_vtk.InsertNextPoint(points[i])
    points_array = npvtk.numpy_to_vtk(np.asarray(points))
    points_vtk.SetData(points_array)
    old_mesh_num_points = points_vtk.GetNumberOfPoints()
    mesh_vtk.SetPoints(points_vtk)

    # Add cells(tris) in to mesh_vtk
    mesh_cell_array = vtk.vtkCellArray()
    mesh_cell_array_np = pv_mesh.cells
    mesh_n_cells = pv_mesh.n_cells
    mesh_cell_array_data= npvtk.numpy_to_vtk(mesh_cell_array_np, array_type=vtk.VTK_ID_TYPE)
    mesh_cell_array.SetCells(mesh_n_cells,mesh_cell_array_data) #this is the legacy way
    mesh_vtk.SetCells(5,mesh_cell_array)
    
    # now we extract edges:
    #BTW lets transfer unstructured grid to polydata in vtk first
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(mesh_vtk)
    geometryFilter.Update()
    mesh_vtp = geometryFilter.GetOutput()

    # assign ids to points using vtkfilter
    idFilter = vtk.vtkIdFilter()
    idFilter.SetInputData(mesh_vtp)
    # This is depreciated in vtk<=9.0.0
    # idFilter.SetIdsArrayName("ids")
    # Available for vtk>=8.3, should use if vtk>9.0.0
    idFilter.SetPointIdsArrayName("ids")
    idFilter.SetPointIds(True)
    idFilter.SetCellIds(False)
    idFilter.Update()
    mesh_vtp_labelid = idFilter.GetOutput()

    # extract edges
    edges_ext_tool = vtk.vtkFeatureEdges()
    edges_ext_tool.SetInputConnection(idFilter.GetOutputPort())
    edges_ext_tool.BoundaryEdgesOn()
    edges_ext_tool.ManifoldEdgesOff()
    edges_ext_tool.NonManifoldEdgesOff()
    edges_ext_tool.FeatureEdgesOff()
    edges_ext_tool.Update()
    edges = edges_ext_tool.GetOutput()
    edge_array = edges.GetPointData().GetArray("ids")
    edges_n_segs = edges.GetNumberOfPoints()
    edge_array_np = []
    for i in range(edges_n_segs):
        edge_array_np.append(edge_array.GetValue(i))

    # use vtk strip to combine segments
    edges.GetNumberOfCells() 
    strip = vtk.vtkStripper()
    strip.SetInputData(edges)
    strip.Update()
    merged_edges = strip.GetOutput()
    merged_edges_array = merged_edges.GetPointData().GetArray("ids")
    merged_edges_n_segs = merged_edges.GetNumberOfPoints()
    merged_edges_array_list = []
    for i in range(merged_edges_n_segs):
        merged_edges_array_list.append(merged_edges_array.GetValue(i))
    merged_edges_array_np = np.asarray(merged_edges_array_list) #mesh pt ids in edge id seq
    merged_edges_cell_array_np = npvtk.vtk_to_numpy(merged_edges.GetLines().GetData()) ##edge id connectivity
    # using array and cell_array to sort meshptid in consecutive seq that forms polyline edge.
    merged_edges_array_np_sorted = np.zeros_like(merged_edges_array_np)
    for i in range(len(merged_edges_array_np)):
        ii = i+1
        merged_edges_array_np_sorted[-i]=merged_edges_array_np[merged_edges_cell_array_np[ii]] # the ext feature is counter clockwise, i need clockwise
        merged_edges_array_np_sorted = np.asarray(merged_edges_array_np_sorted)
    
    #break the polyline connectivity using first point of the outline 
    break_head = np.where(merged_edges_array_np_sorted==0)[0]
    merged_edges_array_np_sorted_rehead = pm.change_polyline_head(merged_edges_array_np_sorted,break_head)
    merged_edges_array_np_sorted_rehead_loop = np.hstack((merged_edges_array_np_sorted_rehead,0))
 
    # break the sorted lines into four segs: inlet, tw, outlet, bw. clockwise fashion
    seg_density = [resr,resx,resr,resx]
    seg_name = ['inlet','topWall','outlet','bottomWall']
    break_num = 4
    break_index = np.zeros(break_num+1,np.int16) # break index in 1,2,3,4,5,.. seq
    break_index_inmesh = np.zeros(break_num+1,np.int16) #break index in mesh ptid seq

    segs_index_list = []
    for i in range(break_num):
        break_index[i+1] = np.cumsum(seg_density)[i]-(i+1)
        if i == break_num-1:
            break_index_inmesh[i+1] = len(merged_edges_array_np_sorted_rehead) #the seq is [pt0,pt1,pt2,pt3,pt4,pt1]
        else:
            break_index_inmesh[i+1] = np.where(merged_edges_array_np_sorted_rehead==break_index[i+1])[0]

        segs_index_list.append([break_index_inmesh[i],break_index_inmesh[i+1]+1])
    # print(segs_index_list)
    segs_index_new = []
    for i in range(break_num):
        segs_index_new.append(merged_edges_array_np_sorted_rehead_loop[segs_index_list[i][0]:segs_index_list[i][1]])
        
    
    # now we rebuild the mesh by constructing a vtk mesh from scratch. 
    # first prepare all the data/nparrays
    # points:
    top_points = pv_mesh.points.copy()
    
    top_points[:,2] = +0.5  # move top plane up
    top_connection = pv_mesh.cells_dict[5]
    top_connection_flipz = top_connection.copy() # switch normal from up to down 
    top_connection_flipz[:,[1,2]] = top_connection_flipz[:,[2,1]] # switch normal from up to down 

    bottom_points = pv_mesh.points.copy() # get tri connection
    bottom_points[:,2] = -0.5 #move bottom plane down
    bottom_connection = pv_mesh.cells_dict[5]
    bottom_connection_flipz = bottom_connection.copy() # switch normal from up to down 
    bottom_connection_flipz[:,[1,2]] = bottom_connection_flipz[:,[2,1]] # switch normal from up to down 
    bottom_connection += len(top_points) #offset the indexs of bottom points by num of points in top plane
    bottom_connection_flipz += len(top_points) #offset the indexs of bottom points by num of points in top plane
    
    mesh_final_num_points = len(top_points)+len(bottom_points)

    # segs_index_new has all the points index of boundaries in sequence!!!
    # In the segs, we have: inlet_index=0,uw_index=1,outlet_index=2,lw_index =3
    boundary_connection = []
    for i in range(break_num): 
        boundary_connection.append(np.column_stack((segs_index_new[i][:-1],segs_index_new[i][1:],segs_index_new[i][1:]+len(top_points),segs_index_new[i][:-1]+len(top_points))))
    fbplane_connection=np.vstack((top_connection,bottom_connection))
    cell_connection=np.column_stack((top_connection,bottom_connection))
    # print(top_connection[0])
    # print(top_points[[top_connection[0][0],top_connection[0][1],top_connection[0][2]],:])
    #creat summary for vtk and vtp file. summary is long, short is to save to json file
    cell_summary = {}
    cell_summary_short = {}
    cell_summary[0] = [13,6,cell_connection,'volume','wedge']
    cell_summary_short[0] = [len(cell_connection),'volume','wedge']
    for i in range(break_num):
        cell_summary[i+1] = [9,4,boundary_connection[i],seg_name[i],'quad']
        cell_summary_short[i+1] = [len(boundary_connection[i]),seg_name[i],'quad']
    cell_summary[break_num+1] =[5,3,fbplane_connection,'frontAndBackPlanes','triangle']
    cell_summary_short[break_num+1] =[len(fbplane_connection),'frontAndBackPlanes','triangle']

    # attension, only boundary in the label list
    cell_label_list = []
    for i in range(len(cell_summary)-1):
        cell_label_list+= [i+1]*len(cell_summary[i+1][2])

    #Start building mesh, here we build two mesh, vtk and vtp. vtk is for volume only, vtp is for surface mesh only!!
    # first we set up points for both vtp and vtk file because they have to have same point id. 
    points_final = vtk.vtkPoints()
    points_final_array = npvtk.numpy_to_vtk(np.vstack((top_points,bottom_points)))
    points_final.SetData(points_final_array)
    mesh_final_num_points = points_final.GetNumberOfPoints()

    # now create vtkfile!
    #set points for vtk
    mesh_final_vtk = vtk.vtkUnstructuredGrid()
    mesh_final_vtk.SetPoints(points_final)
    
    for i in range(len(cell_summary[0][2])):
        mesh_final_vtk.InsertNextCell(cell_summary[0][0],cell_summary[0][1],tuple(cell_summary[0][2][i]))

    # now create vtpfile!
    mesh_final_vtp = vtk.vtkPolyData()
    mesh_final_vtp.SetPoints(points_final)

    # now we use insert next cell to push all of the elements in:
    # specify num of faces you have to insert first using allocate 
    num_of_2d_cells = np.sum([cell_summary_short[ele+1][0] for ele in range(len(cell_summary_short)-1)])
    mesh_final_vtp.Allocate(num_of_2d_cells)
    
    
    #inserting
    for i in range(len(cell_summary)-1):
        #fill inlet,tw,outlet,bw,fp,bp 
        for ii in range(len(cell_summary[i+1][2])):
            mesh_final_vtp.InsertNextCell(cell_summary[i+1][0],cell_summary[i+1][1],tuple(cell_summary[i+1][2][ii]))
    
    #surface elements are finished, now insert boundary condition 
    cell_label_np = np.array(cell_label_list,dtype=np.int32)
    cell_label = npvtk.numpy_to_vtk(cell_label_np, deep=1)
    cell_label.SetName('FaceEntityIds')
    mesh_final_vtp.GetCellData().AddArray(cell_label)

    #set point_data
    point_label_np = np.arange(mesh_final_num_points)  #this is ptid stype
    # point_label_np = np.repeat(3,mesh_final_num_points)  #this is all ptid=3
    point_label_np = np.array(point_label_np,dtype=np.int32)
    point_label = npvtk.numpy_to_vtk(point_label_np, deep=1)
    point_label.SetName('PointEntityids')
    mesh_final_vtk.GetPointData().AddArray(point_label)
    mesh_final_vtp.GetPointData().AddArray(point_label)


    # writing data
    if write != False:
        # volume mesh
        vtkwriter = vtk.vtkUnstructuredGridWriter()
        vtkwriter.SetFileName(os.path.join(save_dir,file_name))
        vtkwriter.SetInputData(mesh_final_vtk)
        vtkwriter.SetFileVersion(version)
        vtkwriter.Update()
        #surface mesh
        vtpwriter = vtk.vtkXMLPolyDataWriter()
        vtpwriter.SetFileName(os.path.join(save_dir,file_name[:-4]+'.vtp'))
        vtpwriter.SetInputData(mesh_final_vtp)
        vtpwriter.Update()


    #writing cell summary dict
    with open(os.path.join(save_dir,file_name[:-4]+'.json'),'w') as f:
        json.dump(cell_summary_short,f)

    return mesh_final_vtk,mesh_final_vtp,cell_summary,cell_summary_short


def triangulate_2d(aorta,mesh_size,num_lloyd_steps=10, write=False, save_dir=None, file_name = None, version =42, write_vtp=True):
    #randomseed
    # np.random.seed(0)
    # read res from aorta
    resr,resx = len(aorta['inlet_spline']),len(aorta['uw_spline'])
    # use pygalmesh to triangulate
    points_outline = aorta['outline']
    constraints = np.column_stack([np.arange(0,len(points_outline)),np.arange(1,len(points_outline)+1)])%len(points_outline)
    mesh = pygalmesh.generate_2d(points_outline[:,:2],constraints,max_edge_size=mesh_size,num_lloyd_steps=10)
    # pass the tri surface to pyvista
    pv_mesh = pv.wrap(mesh)
    # get edge ids
    # Add points in to mesh_vtk
    points=pv_mesh.points
    points_vtk = vtk.vtkPoints()
    mesh_vtk = vtk.vtkUnstructuredGrid()
    for i in range(len(points)):
        points_vtk.InsertNextPoint(points[i])
    points_array = npvtk.numpy_to_vtk(np.asarray(points))
    points_vtk.SetData(points_array)
    old_mesh_num_points = points_vtk.GetNumberOfPoints()
    mesh_vtk.SetPoints(points_vtk)

    # Add cells(tris) in to mesh_vtk
    mesh_cell_array = vtk.vtkCellArray()
    mesh_cell_array_np = pv_mesh.cells
    mesh_n_cells = pv_mesh.n_cells
    mesh_cell_array_data= npvtk.numpy_to_vtk(mesh_cell_array_np, array_type=vtk.VTK_ID_TYPE)
    mesh_cell_array.SetCells(mesh_n_cells,mesh_cell_array_data) #this is the legacy way
    mesh_vtk.SetCells(5,mesh_cell_array)
    
    # now we extract edges:
    #BTW lets transfer unstructured grid to polydata in vtk first
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(mesh_vtk)
    geometryFilter.Update()
    mesh_vtp = geometryFilter.GetOutput()

    # assign ids to points using vtkfilter
    idFilter = vtk.vtkIdFilter()
    idFilter.SetInputData(mesh_vtp)
    # This is depreciated in vtk<=9.0.0
    # idFilter.SetIdsArrayName("ids")
    # Available for vtk>=8.3, should use if vtk>9.0.0
    idFilter.SetPointIdsArrayName("ids")
    idFilter.SetPointIds(True)
    idFilter.SetCellIds(False)
    idFilter.Update()
    mesh_vtp_labelid = idFilter.GetOutput()

    # extract edges
    edges_ext_tool = vtk.vtkFeatureEdges()
    edges_ext_tool.SetInputConnection(idFilter.GetOutputPort())
    edges_ext_tool.BoundaryEdgesOn()
    edges_ext_tool.ManifoldEdgesOff()
    edges_ext_tool.NonManifoldEdgesOff()
    edges_ext_tool.FeatureEdgesOff()
    edges_ext_tool.Update()
    edges = edges_ext_tool.GetOutput()
    edge_array = edges.GetPointData().GetArray("ids")
    edges_n_segs = edges.GetNumberOfPoints()
    edge_array_np = []
    for i in range(edges_n_segs):
        edge_array_np.append(edge_array.GetValue(i))

    # use vtk strip to combine segments
    edges.GetNumberOfCells()
    strip = vtk.vtkStripper()
    strip.SetInputData(edges)
    strip.Update()
    merged_edges = strip.GetOutput()
    merged_edges_array = merged_edges.GetPointData().GetArray("ids")
    merged_edges_n_segs = merged_edges.GetNumberOfPoints()
    merged_edges_array_list = []
    for i in range(merged_edges_n_segs):
        merged_edges_array_list.append(merged_edges_array.GetValue(i))
    merged_edges_array_np = np.asarray(merged_edges_array_list) #mesh pt ids in edge id seq
    merged_edges_cell_array_np = npvtk.vtk_to_numpy(merged_edges.GetLines().GetData()) ##edge id connectivity
    # using array and cell_array to sort meshptid in consecutive seq that forms polyline edge.
    merged_edges_array_np_sorted = np.zeros_like(merged_edges_array_np)
    for i in range(len(merged_edges_array_np)):
        ii = i+1
        merged_edges_array_np_sorted[-i]=merged_edges_array_np[merged_edges_cell_array_np[ii]] # the ext feature is counter clockwise, i need clockwise
        merged_edges_array_np_sorted = np.asarray(merged_edges_array_np_sorted)
    
    #break the polyline connectivity using first point of the outline 
    break_head = np.where(merged_edges_array_np_sorted==0)[0]
    merged_edges_array_np_sorted_rehead = pm.change_polyline_head(merged_edges_array_np_sorted,break_head)
    merged_edges_array_np_sorted_rehead_loop = np.hstack((merged_edges_array_np_sorted_rehead,0))
 
    # break the sorted lines into four segs: inlet, tw, outlet, bw. clockwise fashion
    seg_density = [resr,resx,resr,resx]
    seg_name = ['inlet','topWall','outlet','bottomWall']
    break_num = 4
    break_index = np.zeros(break_num+1,np.int16) # break index in 1,2,3,4,5,.. seq
    break_index_inmesh = np.zeros(break_num+1,np.int16) #break index in mesh ptid seq

    segs_index_list = []
    for i in range(break_num):
        break_index[i+1] = np.cumsum(seg_density)[i]-(i+1)
        if i == break_num-1:
            break_index_inmesh[i+1] = len(merged_edges_array_np_sorted_rehead) #the seq is [pt0,pt1,pt2,pt3,pt4,pt0]
        else:
            break_index_inmesh[i+1] = np.where(merged_edges_array_np_sorted_rehead==break_index[i+1])[0]

        segs_index_list.append([break_index_inmesh[i],break_index_inmesh[i+1]+1])
    # print(segs_index_list)
    segs_index_new = []
    for i in range(break_num):
        segs_index_new.append(merged_edges_array_np_sorted_rehead_loop[segs_index_list[i][0]:segs_index_list[i][1]])
        
    
    # now we rebuild the mesh by constructing a vtk mesh from scratch. 
    # first prepare all the data/nparrays
    # points:
    surf_points = pv_mesh.points.copy()
    surf_connection = pv_mesh.cells_dict[5]
    #total point numbers
    mesh_final_num_points = len(surf_points)

    # segs_index_new has all the points index of boundaries in sequence!!!
    # In the segs, we have: inlet_index=0,uw_index=1,outlet_index=2,lw_index =3
    boundary_connection = []
    for i in range(break_num): 
        boundary_connection.append(np.column_stack((segs_index_new[i][:-1],segs_index_new[i][1:])))
    #creat summary for vtk and vtp file. summary is long, short is to save to json file
    cell_summary = []
    cell_summary_short = []
    cell_summary.append([5,3,len(surf_connection),surf_connection,'interior','triangle'])
    cell_summary_short.append([5,3,len(surf_connection),'interior','triangle'])
    for i in range(break_num):
        cell_summary.append([3,2,len(boundary_connection[i]),boundary_connection[i],seg_name[i],'line'])
        cell_summary_short.append([3,2,len(boundary_connection[i]),seg_name[i],'line'])
    cell_summary.append([0,1,len(surf_points),surf_points,'points','vertex'])
    cell_summary_short.append([0,1,len(surf_points),'points','vertex'])



    # attension, only boundary in the label list
    cell_label_list = []
    for i in range(len(cell_summary)-1):
        cell_label_list+= [i+1]*cell_summary[i+1][2]

    #Start building mesh, here we build only vtp. vtp is for surface mesh only!!
    # first we set up points for both vtp and vtk file because they have to have same point id. 
    points_final = vtk.vtkPoints()
    points_final_array = npvtk.numpy_to_vtk(surf_points)
    points_final.SetData(points_final_array)
    mesh_final_num_points = points_final.GetNumberOfPoints()

    # now create vtpfile!
    mesh_final_vtp = vtk.vtkPolyData()
    mesh_final_vtp.SetPoints(points_final)

    # now we use insert next cell to push all of the elements in:
    # specify num of faces you have to insert first using allocate 
    num_of_2d_cells = cell_summary_short[0][2]
    # print(num_of_2d_cells)
    mesh_final_vtp.Allocate(num_of_2d_cells)
    
    
    #inserting
    for i in range(num_of_2d_cells):
        mesh_final_vtp.InsertNextCell(cell_summary[0][0],cell_summary[0][1],tuple(cell_summary[0][3][i]))
    
    #set point_data
    point_label_np = np.arange(mesh_final_num_points)  #this is ptid stype
    # point_label_np = np.repeat(3,mesh_final_num_points)  #this is all ptid=3
    point_label_np = np.array(point_label_np,dtype=np.int32)
    point_label = npvtk.numpy_to_vtk(point_label_np, deep=1)
    point_label.SetName('PointEntityids')
    mesh_final_vtp.GetPointData().AddArray(point_label)

    # writing data
    if write != False:
        #surface mesh
        vtpwriter = vtk.vtkXMLPolyDataWriter()
        vtpwriter.SetFileName(os.path.join(save_dir,file_name[:-4]+'.vtp'))
        vtpwriter.SetInputData(mesh_final_vtp)
        vtpwriter.Update()       


    #writing cell summary dict
    with open(os.path.join(save_dir,file_name[:-4]+'.json'),'w') as f:
        json.dump(cell_summary_short,f)

    return mesh_final_vtp,cell_summary,cell_summary_short

def vtktosu2(sum1,output_dir, patch_name =None, scale=1):

    # Store the number of nodes and open the output mesh file
    Npts = sum1[-1][2] # total number of points 
    Nelem  = sum1[0][2] # total number of elements
    Nbound = len(sum1)-2 # total number of boundaries
    Mesh_File = open(os.path.join(output_dir, "aorta_2d.su2"),"w")

    # Write the dimension of the problem and the number of interior elements
    Mesh_File.write( "%\n" )
    Mesh_File.write( "% Problem dimension\n" )
    Mesh_File.write( "%\n" )
    Mesh_File.write( "NDIME= 2\n" )
    Mesh_File.write( "%\n" )
    Mesh_File.write( "% Inner element connectivity\n" )
    Mesh_File.write( "%\n" )
    Mesh_File.write( "NELEM= %s\n" % (Nelem))
    # Write the element connectivity
    for i in range(Nelem):
        Mesh_File.write( "%s \t %s \t %s \t %s \t %s\n" % (sum1[0][0], sum1[0][3][i,0],sum1[0][3][i,1],sum1[0][3][i,2], i) )
    # Write the points
    Mesh_File.write( "%\n" )
    Mesh_File.write( "% Node coordinates\n" )
    Mesh_File.write( "%\n" )
    Mesh_File.write( "NPOIN= %s\n" % (Npts) )
    for i in range(Npts):
        Mesh_File.write( "%15.14f \t %15.14f \t %s\n" % (sum1[-1][3][i,0]*scale, sum1[-1][3][i,1]*scale, i) )

    # Write the header information for the boundary markers
    Mesh_File.write( "%\n" )
    Mesh_File.write( "% Boundary elements\n" )
    Mesh_File.write( "%\n" )
    Mesh_File.write( "NMARK= %d\n" % (Nbound) )


    # Write the boundary information for each marker
    for i in range(Nbound):
        Mesh_File.write( "MARKER_TAG= %s\n" %(sum1[i+1][-2]))
        Mesh_File.write( "MARKER_ELEMS= %s\n" % (sum1[i+1][2]))
        for j in range(sum1[i+1][2]):
            Mesh_File.write( "%s \t %s \t %s\n" % (sum1[i+1][0], sum1[i+1][3][j,0], sum1[i+1][3][j,1]) )

    # Close the mesh file and exit
    Mesh_File.close()


def write_inlet_file(sum1, profile=None,output_dir = './',scale=1): 
    #compute normal 
    points = sum1[-1][3]*scale # please remember to scale
    start_pts = points[sum1[1][3][0,0]]
    end_pts = points[sum1[1][3][-1,-1]]
    inlet = end_pts-start_pts  # clock wise 
    mid_point = (start_pts+end_pts)/2 
    normal =np.array([-inlet[1],inlet[0],0])  # conterclockwise 90 from inlet vector
    radius = np.linalg.norm(normal)/2
    norm_normal = normal/(2*radius)
    inlet_normal = inlet/(2*radius)
    
    ptsin = []
    ptsdin =[]
    ptsin.append(points[sum1[1][3][0,0]])
    ptsdin.append(0)

    for i in range(sum1[1][2]):
        d = np.linalg.norm(points[sum1[1][3][i,1]]-start_pts)
        ptsin.append(points[sum1[1][3][i,1]])
        ptsdin.append(d)
     
    
    xx = np.linspace(0,1,len(profile))
    
    print(profile.shape)
    if [profile] == None: 
        Vm=1
        parabolic = lambda x : -Vm/(radius**2)*(x-radius)**2+Vm 
        yy_y = -parabolic(xx)
        yy_x = -np.zeros(len(profile))
    else: 
        # print(profile[:,0])
        yy_x = -np.interp(ptsdin/ptsdin[-1], xx, profile[:,0])
        yy_y = -np.interp(ptsdin/ptsdin[-1], xx, profile[:,1])
    
    v_mag = []
    v_vec = []
    for i in range(sum1[1][2]+1):
        temp_v = yy_y[i]*norm_normal+yy_x[i]*inlet_normal
        v_mag.append(np.linalg.norm(temp_v))
        v_vec.append(temp_v/np.linalg.norm(temp_v))
    v_mag= np.array(v_mag)
    v_vec= np.array(v_vec)
    
    #writing inlet file
    Mesh_File = open(os.path.join(output_dir, "inlet.dat"),"w")

    # Write the dimension of the problem and the number of interior elements
    Mesh_File.write( "NMARK= %d\n" % (1) )
    Mesh_File.write( "MARKER_TAG= %s\n" %(sum1[1][-2]))
    Mesh_File.write( "NROW= %s\n" %(sum1[1][2]+1))
    Mesh_File.write( "NCOL= %s\n" % (6))
    for i in range(len(ptsin)):
        Mesh_File.write( "%15.14f \t %15.14f \t %15.14f \t %15.14f \t %15.14f \t %15.14f\n"
        % (ptsin[i][0],ptsin[i][1],1,v_mag[i],v_vec[i,0],v_vec[i,1]) )

    # Close the mesh file and exit
    Mesh_File.close()



def probe(x,path):
    file_header = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  5.x                                   |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
-------------------------------------------------------------------------------
Description
    Writes out values of fields interpolated to a specified cloud of points.
\*---------------------------------------------------------------------------*/
"""

    bottom_separator = """
\n// ************************************************************************* //
"""
    probe_file = open(os.path.join(path, "internalCloud"), "w")
    probe_file.write(file_header)
    probe_file.write("\nfields (U);\npoints\n(\n")
    for item in x:
        probe_file.write("    (%f %f 0.0)\n" % (item[0],item[1]) )
    probe_file.write("\n);\n\n#includeEtc \"caseDicts/postProcessing/graphs/sampleDict.cfg\"\n#includeEtc \"caseDicts/postProcessing/probes/internalCloud.cfg\"\n")
    probe_file.write(bottom_separator)
    probe_file.close()