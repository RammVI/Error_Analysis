
# 18.08.19
# Updates : Use of reaction potential instead of using the harmonic and regular component

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
#from bem_parameters import *

global mol_name , mesh_density , suffix , path , q , x_q , phi_space , phi_order , u_space , u_order

def main(name , dens , input_suffix , output_suffix , percentaje , smooth = False , refine = True):
    
    init_time = time.time()

    mesh_info.mol_name     = name
    mesh_info.mesh_density = dens
    mesh_info.suffix       = input_suffix
    mesh_info.path         = os.path.join('Molecule' , mesh_info.mol_name)

    mesh_info.q , mesh_info.x_q = run_pqr(mesh_info.mol_name)

    mesh_info.u_space , mesh_info.u_order     = 'DP' , 0
    mesh_info.phi_space , mesh_info.phi_order = 'P' , 1
    mesh_info.u_s_space , mesh_info.u_s_order = 'P' , 1


    bempp.api.set_ipython_notebook_viewer()
    bempp.api.PLOT_BACKEND = "ipython_notebook"

    grid = Grid_loader( mesh_info.mol_name , mesh_info.mesh_density , mesh_info.suffix )

    dirichl_space_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order)
    neumann_space_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order) 
    dual_to_dir_s_u = bempp.api.function_space(grid,  mesh_info.u_space, mesh_info.u_order)

    #u_s  = bempp.api.GridFunction(dirichl_space_u, fun=u_s_G )
    #du_s = bempp.api.GridFunction(neumann_space_u, fun=du_s_G)

    #u_h , du_h = harmonic_component(dirichl_space_u , neumann_space_u , dual_to_dir_s_u , u_s , du_s)
    #u_r , du_r = regular_component(dirichl_space_u , neumann_space_u , dual_to_dir_s_u , du_s , du_h)
    U, dU , U_R , dU_R = U_and_U_Reac(dirichl_space_u , neumann_space_u , dual_to_dir_s_u )
    
    S_trad = S_trad_calc_R( dirichl_space_u, neumann_space_u , U , dU )
    #S_trad_calc( dirichl_space_u, neumann_space_u , u_h , du_h , u_r , du_r)
    
    endt_time = time.time()
    fin_time = endt_time - init_time
    
    ref_init_time = time.time()

    dirichl_space_phi = bempp.api.function_space(grid,  mesh_info.phi_space , mesh_info.phi_order)
    neumann_space_phi = bempp.api.function_space(grid,  mesh_info.phi_space , mesh_info.phi_order) 
    dual_to_dir_s_phi = bempp.api.function_space(grid,  mesh_info.phi_space , mesh_info.phi_order)

    phi , dphi = adjoint_equation( dirichl_space_phi , neumann_space_phi , dual_to_dir_s_phi)

    aux_path = '_'+str(mesh_info.mesh_density)+ mesh_info.suffix

    face_array = np.transpose(grid.leaf_view.elements)+1
    vert_array = np.transpose(grid.leaf_view.vertices)
    
    S_Cooper , S_Cooper_i = S_Cooper_calc( face_array , vert_array , phi , dphi , U_R , dU_R , 25)
    
    dirichl_space_u_s  = bempp.api.function_space(grid,  mesh_info.u_s_space , mesh_info.u_s_order )
    neumann_space_du_s = bempp.api.function_space(grid,  mesh_info.u_s_space , mesh_info.u_s_order )

    u_s  = bempp.api.GridFunction(dirichl_space_u_s , fun=u_s_G )
    du_s = bempp.api.GridFunction(neumann_space_du_s, fun=du_s_G)      
    
    S_Zeb    , S_Zeb_i    = Zeb_aproach_with_u_s_Teo( face_array , vert_array , phi , dphi , 25)
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
            fine_vert_array = text_to_list(name , '_40.0-0' , '.vert' , info_type=float)
            new_vert_array = smoothing_vertex( new_vert_array , fine_vert_array )

        vert_and_face_arrays_to_text_and_mesh( name , new_vert_array ,
                                            new_face_array.astype(int)[:] , output_suffix, dens , Self_build=True)

        grid = Grid_loader( name , dens , output_suffix )
        print('New mesh:')
        #grid.plot()
    
    N_elements = len(face_array)
    
    ref_fin_time = endt_time - ref_init_time
    
    return S_trad , S_Cooper , S_Zeb , N_elements , fin_time , ref_fin_time
    #return dif[:,0]
    
#print('Test')

if True:
    Resultados = open( 'Resultados_20-08-19.txt' , 'w+' )

    Resultados.write( ' density  & Times-Refinated & Num of Elem  & Strad & SCooper & SZeb & time \n' )

    for molecule in ('methanol' , 'arg'):

        for dens in (2.0 , 4.0 , 5.0 , 8.0 , 12.5):

            c_r = 0
            instance = ('-0' , '-1' , '-2' , '-3' , '-4', '-5' , '-6') #, '-7' , '-8' , '-9' , '-10')
            for suffix in instance[:-1]:
                # Refinning 3 times
                text = '{0} & {1}'.format( str(dens) , suffix  )

                S_trad , S_Cooper , S_Zeb , N_elements, fin_time , ref_time = main( molecule , dens , instance[c_r] , instance[c_r+1] , 0.10)

                text = text + ' & {0:d} & {1:.10f} & {2:.10f} & {3:.10f} & {4:.10f}  {4:.10f}\n'.format( N_elements , S_trad , 
                                                                                   S_Cooper , S_Zeb , fin_time, ref_time)
                Resultados.write( text )
                c_r+=1
    Resultados.close()
    
if False:
    
    Resultados = open( 'Resultados_18-08-19.txt' , 'w+' )
    
    Resultados.write( ' density  & Times-Refinated & Num of Elem  & Strad & S_Aprox & S_Ex & time  \n' )
    
    dens = 5.0
    
    
    instance = ('-0' , '-1' , '-2' , '-3' , '-4' , '-5' , '-6' , '-7' , '-8' , '-9' , '-10') #,
#               '-11' , '-12' , '-13' , '-14' , '-15' , '-16' , '-17' , '-18' , '-19' , '-20') 
    c_r = 0
    
    for molecule in ('methanol' , 'arg'):
        
        c_r = 0
        for suffix in instance[:-1]:

            text = '{0} & {1}'.format( str(dens) , suffix  )

            S_trad , S_Cooper , S_Zeb , N_elements, fin_time = main( molecule , dens , suffix , instance[c_r+1] , 0.05)

            text = text + '&{0:d}&{1:.10f}&{2:.10f}&{3:.10f}&{4:.10f}\n'.format( N_elements , S_trad , 
                                                                                       S_Cooper , S_Zeb , fin_time)

            Resultados.write( text )

            c_r += 1
    
    Resultados.close()

if False:
    Resultados = open( 'Resultados_05-07-19.txt' , 'w+' )
    
    #for molecule in ('methanol' , 'arg'):
    if True:
        molecule = 'methanol'
        for dens in (2.0 , 3.0 ,5.0 , 6.0 , 7.0 , 9.0 ): #, 4.0 , 5.0 , 6.0 , 8.0 , 9.0 ):
    
            # Refinning 3 times
            text = '{0} & {1}'.format( str(dens) , suffix  )
            smoothed = False

            S_trad , S_Cooper , S_Zeb , N_elements = main( molecule , dens , '-0' , '' , 0.05 , refine = False)

            text = text + ' & {0:d} & {1:.10f} & {2:.10f} & {3:.10f} \n'.format( N_elements , S_trad , 
                                                                                   S_Cooper , S_Zeb)
            Resultados.write( text )

            
    Resultados.close()
    
if False:
    
    Resultados = open( 'Resultados_arg.txt' , 'w+' )
    
    molecule = 'arg'
    dens = 2.0
        
    instance = ( '-0' , '-R1' , '-R2' , '-R3' )
    
    c_i = 0
    for i in instance[:-1]:
        text = '{0} & {1}'.format( str(dens) , suffix  )
    
    
        text = '{0} & {1}'.format( str(dens) , suffix  )
        smoothed = False
        S_trad , S_Cooper , S_Zeb , N_elements , obs_time =  main( molecule , dens , instance[c_i]  , instance[c_i+1] , 2.5)

        
        text = text + ' & {0:d} & {1:.10f} & {2:.10f} & {3:.10f} & {4:10f}\n'.format( N_elements , S_trad , 
                                                                                   S_Cooper , S_Zeb , obs_time)
        Resultados.write( text )

        c_i+=1
        
    Resultados.close()

    
    
#main( 'arg' , 2.0 , '-0' , '-s1' , 0.10)