Results of Quantum Simultion are saved in this folder with file names

"QuantumSimulation_{Polarization}_Polarization_T={T}_Order={Order_formula}_t_MAX={int(num_points / 2) * T}_Trotter_steps={num_Trotter_steps}.txt"

where 
-Polarization is the polarization of the Hamiltonia (Circular/Linear),
-T period of Hamiltonian
-Order_formula order of Trotter formula used
-num_points number of points taken in time, so t in [-num_points * T, num_points * T]
-num_Trotter steps is number of Trotter steps
