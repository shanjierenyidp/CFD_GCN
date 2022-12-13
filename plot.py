import numpy as np
import pyvista as pv

def plot_one(mesh, tag, save_fig_name=None):
    # plotting
    pv.global_theme.font.color = 'black' 
    plotter = pv.Plotter(off_screen=True)
    # print('hello')
    # print(np.max(mesh.point_data[r"$U_m$"]))
    data = mesh.point_data['l_'+tag]
    data_max,data_min = np.max(data),np.min(data)
    dargs0 = dict(
        scalars='l_'+tag,
        cmap="jet",
        show_scalar_bar=False,
        # show_scalar_bar=True,
        clim = [data_min,data_max]
    )
    dargs1 = dict(
        scalars='p_'+tag,
        cmap="jet",
        show_scalar_bar=False,
        # show_scalar_bar=True,
        clim = [data_min,data_max]
    )
    dargs2 = dict(
        scalars='d_'+tag,
        cmap="jet",
        show_scalar_bar=False,
        # show_scalar_bar=True,
        clim = [data_min,data_max]
    )
    dargs3 = dict(
    opacity = 0.0,
    scalars='l_'+tag,
    cmap="jet",
    show_scalar_bar=True,
    # show_scalar_bar=True,
    clim = [data_min,data_max]
    )
    plotter.background_color = 'w'
    # plotter.camera.position = (2.0, 1.0, 10)
    plotter.camera_position = [(+10, 30, 0.55),(+10, 30, -10),(0.0, 10, 0.0)]
    print(plotter.camera_position)
    plotter.camera.zoom(0.12)
    # print('position:', plotter.camera.position)    
    
    plotter.add_mesh(mesh,**dargs0)
    plotter.screenshot(filename=save_fig_name[:-4]+'l_'+tag+save_fig_name[-4:], transparent_background=False, return_img=False, window_size=(800,800))
    plotter.clear()
    plotter.add_mesh(mesh,**dargs1)
    plotter.screenshot(filename=save_fig_name[:-4]+'p_'+tag+save_fig_name[-4:], transparent_background=False, return_img=False, window_size=(800,800))
    plotter.clear()
    plotter.add_mesh(mesh,**dargs2)
    plotter.screenshot(filename=save_fig_name[:-4]+'d_'+tag+save_fig_name[-4:], transparent_background=False, return_img=False, window_size=(800,800))
    plotter.clear()
    plotter.add_mesh(mesh,**dargs3)
    plotter.camera.zoom(1)
    plotter.screenshot(filename=save_fig_name[:-4]+'cmap_'+tag+save_fig_name[-4:], transparent_background=False, return_img=False, window_size=(800,800))
    print('done')

for i in range(1,5):
    # mesh = pv.read('results/test_results_1_2000_200_4/prediction/test{:d}.vtk'.format(i))
    mesh = pv.read('results/test_results_1_2000_3000_4/prediction/test{:d}.vtk'.format(i))
    plot_one(mesh, 'Um', save_fig_name='results/test_results_1_2000_3000_4/prediction/visualize{:d}.png'.format(i))
    plot_one(mesh, 'P', save_fig_name='results/test_results_1_2000_3000_4/prediction/visualize{:d}.png'.format(i))