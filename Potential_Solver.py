
import bempp.api
import numpy as np
import os

from bempp.api.operators.potential import laplace as lp
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

from constants import mesh_info
from constants import values
from constants import *

from quadrature import *



def zero_i(x, n, domain_index, result):
    result[:] = 0

def u_s_G(x,n,domain_index,result):
    result[:] =  C / (4.*np.pi*ep_m)  * np.sum( mesh_info.q / np.linalg.norm( x - mesh_info.x_q, axis=1 ) )

def du_s_G(x,n,domain_index,result):
    result[:] = -1./(4.*np.pi*ep_m)  * np.sum( np.dot( x-
                            mesh_info.x_q , n)  * mesh_info.q / np.linalg.norm( x - mesh_info.x_q, axis=1 )**3 )

def harmonic_component(dirichl_space , neumann_space , dual_to_dir_s , u_s , du_s):
    
    global ep_m , ep_s, k

    # Se crean los operadores asociados al sistema, que dependen de los espacios de las soluciones
    # identity : I  : matriz identidad
    # dlp_in : K_in : Double-Layer operator para la region interior
    # slp_in : V_in : Single-Layer operator para la region interior
    # _out : Mismos operadores pero para la region exterior, con k=kappa=0.125
    identity = sparse.identity(     dirichl_space, dirichl_space, dual_to_dir_s)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dual_to_dir_s)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dual_to_dir_s)

    # V_in du_s = (1/2+K_in)u_h = -(1/2+K_in)u_s (BC)
    sol, info,it_count = bempp.api.linalg.gmres( slp_in, -(dlp_in+0.5*identity)*u_s , return_iteration_count=True, tol=1e-8)
    print("The linear system for du_h was solved in {0} iterations".format(it_count))



    u_h = -u_s
    du_h = sol
    
    return u_h , du_h

def regular_component(dirichl_space , neumann_space , dual_to_dir_s , du_s , du_h):
    
    global ep_m , ep_s, k
    
    identity = sparse.identity(     dirichl_space, dirichl_space, dual_to_dir_s)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dual_to_dir_s)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dual_to_dir_s)
    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dual_to_dir_s, k)
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dual_to_dir_s, k)
    
    # Se crea la matriz / Lado izquierdo de la ecuacion
    # | ( I/2 + K_L-in  )     (      -V_L-in     ) |  u_r  = 0
    # | ( I/2 - K_Y-out )     ( ep_m/ep_s V_Y-out) | du_r  = ep_m/ep_s V_Y-out*(du_s+du_h)  (BC)
    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = 0.5*identity + dlp_in
    blocked[0, 1] = -slp_in
    blocked[1, 0] = 0.5*identity - dlp_out
    blocked[1, 1] = (ep_m/ep_s)*slp_out

    # Se crea el lado derecho de la ecuacion 
    zero = bempp.api.GridFunction(dirichl_space, fun=zero_i)
    rhs = [ zero ,  -slp_out *(ep_m/ep_s)* (du_s+du_h)]

    # Y Finalmente se resuelve para u_r y du_r
    sol, info,it_count = bempp.api.linalg.gmres( blocked , rhs, return_iteration_count=True, tol=1e-8)
    print("The linear system for u_r and du_r was solved in {0} iterations".format(it_count))
    u_r , du_r = sol
    
    return u_r , du_r

def q_times_G_L(x, n, domain_index, result):
    global q,x_q,ep_m,C , k
    result[:] = 1. / (4.*np.pi*ep_m)  * np.sum( mesh_info.q  / np.linalg.norm( x - mesh_info.x_q, axis=1 ) )

def adjoint_equation( dirichl_space , neumann_space , dual_to_dir_s):
    
    global ep_m , ep_s , k
    
    identity = sparse.identity(     dirichl_space, dirichl_space, dual_to_dir_s)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dual_to_dir_s)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dual_to_dir_s)
    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dual_to_dir_s, k)
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dual_to_dir_s, k)

    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = 0.5*identity + dlp_in
    blocked[0, 1] = -slp_in
    blocked[1, 0] = 0.5*identity - dlp_out
    blocked[1, 1] = (ep_m/ep_s)*slp_out

    zero = bempp.api.GridFunction(dirichl_space , fun=zero_i)
    P_GL = bempp.api.GridFunction(dirichl_space, fun=q_times_G_L)
    rs_r = [P_GL , zero]

    sol_r, info,it_count = bempp.api.linalg.gmres( blocked, rs_r , return_iteration_count=True, tol=1e-8)
    print("The linear system for phi was solved in {0} iterations".format(it_count))
    phi_r , dphi_r = sol_r
    
    return phi_r , dphi_r

def S_trad_calc( dirichl_space, neumann_space , u_h , du_h , u_r , du_r):
    
    # En base a los puntos donde se encuentran las cargas, calculemos el potencial u_r y u_h
    # Esto luego de que podemos escribir la energia de solvatacion como
    # G_solv = Sum_i q_i *u_reac = Sum_i q_i * (u_h+u_r)           evaluado en cada carga.

    # Se definen los operadores
    slp_in_O = lp.single_layer(neumann_space, mesh_info.x_q.transpose()) 
    dlp_in_O = lp.double_layer(dirichl_space, mesh_info.x_q.transpose())

    # Y con la solucion de las fronteras se fabrica el potencial evaluada en la posicion de cada carga
    u_r_O = slp_in_O * du_r  -  dlp_in_O * u_r
    u_h_O = slp_in_O * du_h  -  dlp_in_O * u_h

    terms =  u_r_O + u_h_O

    # Donde agregando algunas constantes podemos calcular la energia de solvatacion S
    
    S     = K * np.sum(mesh_info.q * terms).real
    print("Three Term Splitting Solvation Energy : {:7.8f} [kCal/mol] ".format(S) )
    
    return S

def S_Cooper_calc( face_array , vert_array , phi_r , dphi_r , U_Reac , dU_Reac , N):
    
    Solv_Cooper = np.zeros((len(face_array),1))
    c = 0
    for face in face_array:

        I1 = int_calc_i( face , face_array , vert_array , phi_r , mesh_info.phi_space 
                        , mesh_info.phi_order , ep_m*dU_Reac , mesh_info.u_space , mesh_info.u_order  , N)
        
        I2 = int_calc_i( face , face_array , vert_array , ep_m * dphi_r , mesh_info.phi_space
                        , mesh_info.phi_order , U_Reac , mesh_info.u_space , mesh_info.u_order  , N)
        Solv_Cooper[c] = I1-I2

        c+=1

    S_Cooper_i = Solv_Cooper
    S_Cooper = K*np.sum(Solv_Cooper )
    print('Cooper Solv = {0:10f} '.format(S_Cooper)) 
    
    return S_Cooper , S_Cooper_i

def S_Zeb_calc( face_array , vert_array , phi , dphi , u_s , du_s , N):
    
    Solv_Zeb = np.zeros((len(face_array),1))
    c = 0
    for face in face_array:

        I1 = int_calc_i( face , face_array , vert_array , phi , mesh_info.phi_space 
                        , mesh_info.phi_order , ep_m*(du_s) , mesh_info.u_s_space , mesh_info.u_s_order  , N)
        
        I2 = int_calc_i( face , face_array , vert_array ,  dphi , mesh_info.phi_space
                         , mesh_info.phi_order , ep_m*(u_s) , mesh_info.u_s_space , mesh_info.u_s_order , N)
        Solv_Zeb[c] = I2-I1

        c+=1
    Solv_Zeb_i = Solv_Zeb
    S_Zeb = K*np.sum(Solv_Zeb )
    print('Zeb Solv = {0:10f} '.format(S_Zeb)) 
    
    return S_Zeb , Solv_Zeb_i