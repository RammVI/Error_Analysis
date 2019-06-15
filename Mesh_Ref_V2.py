
# Some rules about the mesh refinement:
# 1. A percentaje of the faces must be refined and 4 new triangles are born
#    where 3 new points are added in each edge center, and the input data is the residual
#        .                  ___ .____
#       /\                 |\  /\  /|
#      /  \      Results   | \/__\/ |
#     /    \       in      | /\  /\ |
#    /______\              |/__\/__\|

#     in the solvation energy.
# 2. Adjacent triangles are split but half UNLESS:
#    2.1 They are adjacent to 2 triangles to be refinated into 4 new triangles
# Do this until there is no more triangles in 2.1 .

# Also, the possibility to extrapolate the point to the real boundary will be
# set in a function named new_point()

import bempp.api, numpy as np
from math import pi
import os

def search_unique_position_in_array(array , main_array):
    '''
    -
    '''
    position = -1
    
    c=0
    for sub_array in main_array:
        if (sub_array == array).all():
            break
        c+=1
    return c
    
def search_multiple_positions_in_array( arrays , main_array ):
    '''
    Returns the position of each array (contained in arrays) in main_array.
    Arrays must save the information in rows, for example
    arrays = np.array((f1x , f1y , f1z),
                       f2x , f2y , f2z)....)
    '''    
    
    positions = (-1)*np.ones( (len(arrays),1) )
    
    i=0
    for array in arrays:
        
        c = 0
        
        for sub_array in main_array:
            
            if(array == sub_array).all():
                positions[i] = c                
            
            c+=1
        i+=1
    return positions
        
def text_to_list(mol_name , total_suffix , txt_format , info_type=float):
    '''
    Rutine which builds lists from text files that contain the vert and face info
    mol_name : Abreviated name of the molecule
    total_suffix   : text added after molecule name, for an easier handling of files 
                     may be taken as a total and used like _{density}-{it_count}
    txt_format : file format
    info_type  : float or int
    '''
    path = os.path.join('Molecule',mol_name)
    
    list_txt = open( os.path.join(path , mol_name +total_suffix + txt_format) ).read().split('\n')

    listing = np.empty((0,3))
    for line in list_txt[:-1]:
        info    = np.array(line.split()[:3])
        listing = np.vstack( ( listing , info ) ).astype(info_type)
    return listing


def value_assignor_starter(face_array , soln , percentaje):
    '''
    Assigns face's value to the number of new triangles born (Desirable is 0, 2 or 4).
    May use first!
    '''

    
    # This extract the class value
    separator = np.sort(soln)[int(percentaje * len(soln) )]
        
    refined_faces = np.zeros( (len(soln), ) )
        
    c = 0
    for v in soln:
        
        if v>separator:
            refined_faces[c] = 4
        c+=1
    
    return refined_faces

def coincidence_between_1D_arrays(array1 , array2 , coinc_num ):
    '''
    Search for coincidences between 2 arrays, returns True if arrays have coinc_num of coincidences
    and also returns the coincidences.
    '''
    coincidences = np.zeros( (coinc_num , ) ) 
    
    c=0
    for a1 in array1:
        
        for a2 in array2:
            
            if a1 == a2:
                
                coincidences[c] = a1
                c+=1
                
    if 0 in coincidences:
        return False , np.zeros( (coinc_num , ) )
                
    return True , coincidences

def adjacent_faces(face_num , face_array , return_face_number):
    '''
    Searchs for adjacent faces to be refined and sets value 2 if the face is adjacent to only 1
    face which has 4 new triangles to be refined or 3 if this points has a 2 adj. faces to be refined, and 
    4 if the face has three adj triangles to be refined.
    face_num    : Position of the face in face_array
    face_status : Face position in face_array to be refined into 4 triangles
    return_face_number : Boolean True  if it is required to return the position of the face in face_array.
    '''
    
    adj_faces = np.zeros( (len(face_array), ) )
    
    pointed_face = face_array[face_num]
        
    adj = True
        
    T1 , T2 , T3 = -1 , -1 , -1
        
    cj = 0
        
    for face in face_array:
                        
        if (face == pointed_face).all():
            cj+=1
            continue
            
        Boolean , coincidences = coincidence_between_1D_arrays(pointed_face , face , coinc_num=2 )
        
        if Boolean and T1 == -1:
            T1 = cj
            cj+=1
            continue
        if Boolean and T2 == -1:
            T2 = cj
            cj+=1
            continue
        if Boolean and T3 == -1:
            T3 = cj
            cj+=1
            continue
        
        #if f1 in pointed_face:
                
        #    if (f2 in pointed_face or f3 in pointed_face) and T1 == -1:
        #        T1 = cj
        #        cj+= 1
        #        continue
    
        #if f2 in pointed_face:
                
        #    if (f1 in pointed_face or f3 in pointed_face) and T2 == -1:
        #        T2 = cj
        #        cj+= 1
        #        continue
                
        #if f3 in pointed_face:
        #    print('third cond ', face, pointed_face)
        #    if f1 in pointed_face or f2 in pointed_face:
        #        print('T3: ',face)
        #        T3 = cj
        #        continue
        cj+=1
    
    adj_faces[T1] , adj_faces[T2] , adj_faces[T3] = 2 , 2 , 2 
    
    
    if return_face_number:
        return T1,T2,T3
    
    else:
        return adj_faces

def adj_assigner_value( face_array , soln , percentaje):
    
    first_status = value_assignor_starter(face_array , soln , percentaje)
    
    adj_status   = np.zeros( (len(face_array), ) )
    
    face_num = 0    # face counter
    
    for value in first_status:
        
        if value == 4:
            
            adj1 , adj2 , adj3 = adjacent_faces( face_num , face_array , return_face_number = True)
            
            if adj_status[adj1] != 4:
                adj_status[adj1] +=1
                
            if adj_status[adj2] != 4:
                adj_status[adj2] +=1
                
            if adj_status[adj3] != 4:
                adj_status[adj3] +=1
            
            
        face_num+=1
        
    return adj_status 
        
def final_status(face_array , soln , percentaje ):
    '''
    Runs into status until there is no more triangles splitted in 3 (rule 2.1)
    '''
    
    status = value_assignor_starter(face_array , soln , percentaje )+  \
                adj_assigner_value(face_array, soln , percentaje)
    
    face_num = 0
    aux_status = status.copy()
    for s in status:
        if s >4:
            aux_status[face_num] = 4
        face_num+=1
        
    status = aux_status.copy()        
    
    iteration_restrictor = 0
    
    Calculating = True
    
    while 2 in status and iteration_restrictor<10 or Calculating:
        
        if iteration_restrictor == 9:
            print('Adapting the mesh is taking too long!')
        
        # Changing the 2 values for 4
        
        face_num = 0
        aux_status = status.copy()
        for s in status:
            if s == 2:
                aux_status[face_num] = 4
            face_num += 1
        
        Calculating = True
        
        status = aux_status.copy()
        
        adj_status   = np.zeros( (len(face_array), ) )
        status_4     = np.zeros( (len(face_array), ) )
        
        face_num = 0
        for value in status:
        
            if value == 4:

                adj1 , adj2 , adj3 = adjacent_faces( face_num , face_array , return_face_number = True)

                if status[adj1] != 4:
                    adj_status[adj1] +=1

                if status[adj2] != 4:
                    adj_status[adj2] +=1

                if status[adj3] != 4:
                    adj_status[adj3] +=1
                    
                status_4[face_num] = 4
            face_num+=1

        status = adj_status + status_4  
            
        face_num = 0
        aux_status = status.copy()
        for s in status:
            if s>4:
                aux_status[face_num] = 4
                
            face_num+=1
        
        status = aux_status.copy()
        
        iteration_restrictor += 1
        
        Calculating = False
        
    return aux_status

def funcion(face_array , vert_array , soln , percentaje ):
    '''
    Does the mesh refinement, starting by triangles having status 1.
    '''
    status = final_status(face_array , soln , percentaje )
    
    new_face_array = face_array.copy().astype(int)
    new_vert_array = vert_array.copy()
    
    face_num = 0
    for s in status:
        # 1 values
        if s == 1:
            T1 , T2 , T3 = adjacent_faces(face_num , face_array , return_face_number=True)
            
            # T_i is absolute! ------ starts from 0
            
            if status[T1] == 4:
                adj = face_array[T1]
            elif status[T2] == 4:
                adj = face_array[T2]
            elif status[T3] == 4:
                adj = face_array[T3]
            else:
                print('Fatal error encountered - probably a face is missing')
                return None
            
            
            
            _ , common_verts = coincidence_between_1D_arrays( adj , face_array[face_num] , 2 )
            #print('common_verts = ',common_verts)
            
            
            v1 , v2 = common_verts #- 1        # v1 y v2 are not absolute! - Starts from 1
            
            for v in face_array[face_num]:
                if v != v1 and v!=v2:
                    v3 = v
                    
            
            new_vert = newvert( vert_array[v1-1] , vert_array[v2-1] ) #0.5*(vert_array[v1]+vert_array[v2])            
            
            new_vert_pos = search_unique_position_in_array(new_vert , new_vert_array )
            
            if new_vert_pos == -1:
                
                new_vert_array = np.vstack( ( new_vert_array , new_vert ) )    
                new_vert_pos = len(new_vert_array) 
            
            new_face_array[face_num] = np.zeros( ( 3 , ) )  
            
            print(np.array( (v1 , v3 , new_vert_pos+1 ) ), np.array( (v2 , v3 , new_vert_pos+1 )) )
            
            new_face_array = np.vstack(( new_face_array , np.array( (v1 , v3 , new_vert_pos+1 ) ) ))
            new_face_array = np.vstack(( new_face_array , np.array( (v2 , v3 , new_vert_pos+1 ) ) ))    
            
        # 4 values
        if s == 4:
            
            f1 , f2 , f3 = face_array[face_num]-1   # f is absolute!  - Starts from 0
            v1 , v2 , v3 = vert_array[f1] , vert_array[f2] , vert_array[f3]
            
            v12 = newvert( v1 , v2 )
            v13 = newvert( v1 , v3 )
            v23 = newvert( v2 , v3 )
            
            ck = 0 
            v12pos , v13pos , v23pos = search_multiple_positions_in_array( np.array((v12,v13,v23)),new_vert_array)
            
            if v12pos == -1:
                
                v12pos = len(new_vert_array)
                new_vert_array = np.vstack((new_vert_array , v12 ))
                
            if v13pos == -1:
            
                v13pos = len(new_vert_array)
                new_vert_array = np.vstack((new_vert_array , v13 ))
                
            if v23pos == -1:
                
                v23pos = len(new_vert_array)
                new_vert_array = np.vstack((new_vert_array , v23 ))
            
            v12pos+=1
            v13pos+=1
            v23pos+=1
            
            face1 = np.array((f1+1  , v13pos , v12pos ))
            face2 = np.array((v12pos, v23pos , v13pos ))
            face3 = np.array((v13pos, v23pos , f3 +1  ))
            face4 = np.array((v12pos, v23pos , f2 +1  ))
            
            new_face_array[face_num] = np.zeros( ( 3 , ) )
            
            new_face_array = np.vstack((new_face_array , face1 ))
            new_face_array = np.vstack((new_face_array , face2 ))
            new_face_array = np.vstack((new_face_array , face3 ))
            new_face_array = np.vstack((new_face_array , face4 ))
        
        face_num+=1
        
    aux_face_array = np.empty(( 0,3))
    face_num = 0
    for face in new_face_array:
            
        if not (face.astype(int) == (0,0,0)).all(): #np.zeros( ( 3 , 1) )).all():
            
            print(face)
            aux_face_array = np.vstack( (aux_face_array , face.astype(int) ) )
            
        face_num+=1  
        
    #new_face_array = aux_face_array.copy() # This avoid deleting 0 rows
    return new_face_array.astype(int) , new_vert_array

def newvert(vA,vB):
    return 0.5*(vA+vB)