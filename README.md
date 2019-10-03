# Error Analysis

Error_Analisys_R1 routine solves for the mean electrostatic potential for a single molecule in a continuus domain. Gives 
the error when calculating the Solvation energy. Must add references and some sort of demostration.

Error_2.0.vtk is the error estimation in the solvation energy by element, should be opened with Paraview.

Grid_Maker_R2.py    - Callable library to create the mesh. MSMS allowed

quadrature.py       - functions and quadrature rules to estimate boundary integrals.

constants.py        - info to build the mesh and physical constants

potential_solver.py - Internally builds operands and block matrix to solve the PBE, and the harmonic and regular components.

main.py             - Classical refinement rutine

main_Adj.py         - Estimates the boundary error with a 1-grade uniform refinement.

Mesh_Ref_V2.py      - Functions to refinate some elements, given the face array and vert array

Mesh_Ref.py         - Functions to refinate the mesh, with other method that shall not be used.

file_converter_p2.py- Makes a face and vert array from .off file

bem_parameters.py   - Contains some interesting bempp parameters that are commonly changed

analitycal.py       - Functions to estimate the solvation energy for a sphere geometry with a given charges distribution





