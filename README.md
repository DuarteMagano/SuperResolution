# SuperResolution

Super resolution techniques for quantum simulation.

## Model I: Two-level system driven by Circular Polarized Field

- [ ]  We are working the Hamiltonian of eq 5.1 of the thesis report., for which you’ve calculated the quasi energies. But there is still some confusion about the numerical values… So, write down what are the numerical values for parameters that you are using for the simulation and what are the numerical values of the quasi-energies (and corresponding periods).
- Answer
    - **Description:**
        - 
    - **Questions (for the team):**
        - Q1:
        - Q2:
    - **Answers / discussion:**
        - 
    - **Links / files:**
        - 
- [ ]  Plot the **analytical** signal for the span of several periods (no trotter error, no hardware noise).
- [ ]  Recover the frequencies from DFT and ANM, plotting the reconstruction error as a function of the time window (similar to your plot 5.4.a). Write down explicitly the parameters of your simulation. Be careful choosing the hyperparameter lambda. Make sure that, for large enought times, the error in the DFT converges to that of the ANM. Conclude what is the minimum time  window for which the ANM provides “good” results.
- Answer
    
    
- [ ]  Fixing the time window to the value concluded in the previous point, try out different trotter formulas. You can test first and second orders, and permute the order of the terms. No need to go to a very high number of Trotter steps (this will never be used on hardware). Show the signal as a function of time and the corresponding error. (Do not include hardware noise: the gate decomposition is not relevant at this point)
- Answer
    
    
- [ ]  Do the same as above, but plot the reconstruction error from both the DFT and the ANM.
- Answer
    
    
- [ ]  Repeat the previous two points, but now with some gate error. See qiskit guidelines on how to introduce noise. Repeat for different levels of noise.
- Answer
    
    
- [ ]  Select the Trotter formula that achieves the smallest DFT and ANM reconstruction errors (the best formulas might not be the same for the two methods).
- Answer
    
    
- [ ]  Focusing on those formulas, do a detailed rounded of gate optimization. Ask for help!
- Answer

## Model II: Heisenberg model with three qubits

## Model III: Single-impurity model

## Write paper
