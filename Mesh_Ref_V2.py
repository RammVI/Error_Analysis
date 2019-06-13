
# Some rules about the mesh refinement:
# 1. A percentaje of the faces must be refined and 4 new triangles are born
#    where 3 new points are added in each edge center, and the input data is the residual
#        .                     .
#       /\                    /\
#      /  \      Results     /__\ 
#     /    \       in       /\  /\
#    /______\              /__\/__\

#     in the solvation energy.
# 2. Adjacent triangles are split but half UNLESS:
#    2.1 They are adjacent to 2 triangles to be refinated.
# Do this until there is no more triangles in 2.1 .

# Also, the possibility to extrapolate the point to the real boundary will be
# set in a function named new_point()

import bempp.api, numpy as np
from math import pi
import os

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
            
        f1,f2,f3 = face
            
        if (face == pointed_face).all():
            cj+=1
            continue
                
        if f1 in pointed_face:
                
            if f2 in pointed_face or f3 in pointed_face and T1 == -1:
                
                T1 = cj
                cj+= 1
                continue
    
        if f2 in pointed_face:
                
            if f1 in pointed_face or f3 in pointed_face and T2 == -1:
                    
                T2 = cj
                cj+= 1
                continue
                
        if f3 in pointed_face:
                
            if f1 in pointed_face or f2 in pointed_face:
                T3 = cj
                continue
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