
import bempp.api, numpy as np
from math import pi
import os
import time

from constants import values
from constants import mesh_info

from bempp.api.operators.potential import laplace as lp
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

from Grid_Maker_R2  import *
from Mesh_Ref_V2    import *
from quadrature     import *
from Potential_Solver import *
from sphere_geom    import *
#from bem_parameters import *


global mol_name , mesh_density , suffix , path , q , x_q , phi_space , phi_order , u_space , u_order

def main(name , dens , input_suffix , output_suffix , percentaje , r , x_q
         , q , smooth = False , refine = True ):
    
    init_time = time.time()

    mesh_info.mol_name     = name
    mesh_info.mesh_density = dens
    mesh_info.suffix       = input_suffix
    mesh_info.path         = os.path.join('Molecule' , mesh_info.mol_name)

    mesh_info.q , mesh_info.x_q = q , x_q

    mesh_info.u_space , mesh_info.u_order     = 'DP' , 0
    mesh_info.phi_space , mesh_info.phi_order = 'P' , 1
    mesh_info.u_s_space , mesh_info.u_s_order = 'P' , 1


    bempp.api.set_ipython_notebook_viewer()
    bempp.api.PLOT_BACKEND = "ipython_notebook"

    grid = Grid_loader( mesh_info.mol_name , mesh_info.mesh_density , mesh_info.suffix )

    dirichl_space_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order)
    neumann_space_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order) 
    dual_to_dir_s_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order)

    u_s  = bempp.api.GridFunction(dirichl_space_u, fun=u_s_G )
    du_s = bempp.api.GridFunction(neumann_space_u, fun=du_s_G)

    #u_s.plot()

    u_h , du_h = harmonic_component(dirichl_space_u , neumann_space_u , dual_to_dir_s_u , u_s , du_s)
    u_r , du_r = regular_component(dirichl_space_u , neumann_space_u , dual_to_dir_s_u , du_s , du_h)

    S_trad = S_trad_calc( dirichl_space_u, neumann_space_u , u_h , du_h , u_r , du_r)

    endt_time = time.time()
    fin_time = endt_time - init_time

    dirichl_space_phi = bempp.api.function_space(grid,  mesh_info.phi_space , mesh_info.phi_order)
    neumann_space_phi = bempp.api.function_space(grid,  mesh_info.phi_space , mesh_info.phi_order) 
    dual_to_dir_s_phi = bempp.api.function_space(grid,  mesh_info.phi_space , mesh_info.phi_order)

    phi , dphi = adjoint_equation( dirichl_space_phi , neumann_space_phi , dual_to_dir_s_phi)

    aux_path = '_'+str(mesh_info.mesh_density)+ mesh_info.suffix

    face_array = text_to_list(mesh_info.mol_name , aux_path , '.face' , info_type=int  )
    vert_array = text_to_list(mesh_info.mol_name , aux_path , '.vert' , info_type=float)

    S_Cooper , S_Cooper_i = S_Cooper_calc( face_array , vert_array , phi , dphi , u_r+u_h , du_r+du_h , 25)

    dirichl_space_u_s  = bempp.api.function_space(grid,  mesh_info.u_s_space , mesh_info.u_s_order )
    neumann_space_du_s = bempp.api.function_space(grid,  mesh_info.u_s_space , mesh_info.u_s_order )

    u_s  = bempp.api.GridFunction(dirichl_space_u_s , fun=u_s_G )
    du_s = bempp.api.GridFunction(neumann_space_du_s, fun=du_s_G)      

    S_Zeb    , S_Zeb_i    = S_Zeb_calc( face_array , vert_array , phi , dphi , u_s , du_s , 25)
                                # Zeb_aproach_with_u_s_Teo( face_array , vert_array , phi , dphi , 25)

    const_space = bempp.api.function_space(grid,  "DP", 0)

    Solv_i_Cooper_Func = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Cooper_i[:,0])
    Solv_i_Zeb_Func    = bempp.api.GridFunction(const_space, fun=None, coefficients=S_Zeb_i[:,0])

    dif =S_Cooper_i-S_Zeb_i

    dif_F = bempp.api.GridFunction(const_space, fun=None, coefficients=np.abs(dif[:,0] ) )
    #dif_F.plot()    

    bempp.api.export('Molecule/' + name +'/' + name + '_{0}{1}.vtk'.format( dens,input_suffix )
                         , grid_function = dif_F , data_type = 'element')

    #if False:
    #    status = value_assignor_starter(face_array , np.abs(dif[:,0]) , percentaje) #final_status(face_array , dif[:,0] , 2 )
    #    const_space = bempp.api.function_space(grid,  "DP", 0)
    #    Status    = bempp.api.GridFunction(const_space, fun=None, coefficients=status)
    #    Status.plot()

    if refine:

        new_face_array , new_vert_array = mesh_refiner(face_array , vert_array , np.abs(dif[:,0]) , percentaje )

        if smooth:

            aux_vert_array = np.zeros(( len(new_vert_array),3 ))

            c=0
            for vert in new_vert_array:
                aux_vert_array[c] = smothing_func(vert , r) 
                c+=1

            #print(aux_vert_array)

        vert_and_face_arrays_to_text_and_mesh( name , aux_vert_array ,
                                                new_face_array.astype(int)[:] , output_suffix, dens , Self_build=True)

        grid = Grid_loader( name , dens , output_suffix )
        #print('New mesh:')
        #grid.plot()

        N_elements = len(face_array)
        
    return S_trad , S_Cooper , S_Zeb , N_elements , fin_time

#input_suffix = '-s0'
#output_suffix = '-s1'

N_it = 3
suffixes = suffix_names(N_it)

x_q = np.array( ([0. , 0. , 0. ] , [0. , 0.5 , 0.] ))
q = np.array( (1. , 2.) )

r = 10.
name = 'sphere'
dens = 0
percentaje = 0.5



vert_array , face_array = decahedron_points(r)
bempp.api.set_ipython_notebook_viewer()
bempp.api.PLOT_BACKEND = "ipython_notebook"

vert_and_face_arrays_to_text_and_mesh('sphere' , vert_array , face_array+1 , '-s0' 
                                          , dens=0 , Self_build=True)

grid = Grid_loader('sphere' , 0 , '-s0' )
#grid.plot()

c= 0
for i in suffixes[:-1]:
    
    main(name , dens , suffixes[c] , suffixes[c+1] , percentaje , r , x_q , q , smooth = True)
    
    c+=1
    