# SuperResolution

Super resolution techniques for quantum simulation.

## Model I: Two-level system driven by Circular Polarized Field

- We are working the Hamiltonian of eq 5.1 of the thesis report., for which you’ve calculated the quasi energies. But there is still some confusion about the numerical values… So, write down what are the numerical values for parameters that you are using for the simulation and what are the numerical values of the quasi-energies (and corresponding periods).
- Plot the **analytical** signal for the span of several periods (no trotter error, no hardware noise).
- Recover the frequencies from DFT and ANM, plotting the reconstruction error as a function of the time window (similar to your plot 5.4.a). Write down explicitly the parameters of your simulation. Be careful choosing the hyperparameter lambda. Make sure that, for large enought times, the error in the DFT converges to that of the ANM. Conclude what is the minimum time  window for which the ANM provides “good” results.
- Fixing the time window to the value concluded in the previous point, try out different trotter formulas. You can test first and second orders, and permute the order of the terms. No need to go to a very high number of Trotter steps (this will never be used on hardware). Show the signal as a function of time and the corresponding error. (Do not include hardware noise: the gate decomposition is not relevant at this point)
- Do the same as above, but plot the reconstruction error from both the DFT and the ANM.
- Repeat the previous two points, but now with some gate error. See qiskit guidelines on how to introduce noise. Repeat for different levels of noise.
- Select the Trotter formula that achieves the smallest DFT and ANM reconstruction errors (the best formulas might not be the same for the two methods).
- Focusing on those formulas, do a detailed rounded of gate optimization. Ask for help!

## Model II: Heisenberg model with three qubits

## Model III: Single-impurity model

## Write paper


# Documentation 

## Project structure
- README.md — project overview, usage and documentation.
- requirements.txt - libraries used
- QuantumSimulation.py — lightweight runner / example script for quick tests.
- report.ipynb — notebook with results.
- Files/ — output folder for saved figures, CSVs and experiment artifacts.
- Quantum_Simulation/ — core simulation code:
  - Floquet.py — driven two-level (Floquet) model implementations and helpers.
  - HeisenbergTimeIndependent.py — Heisenberg 3-qubit model and utilities.
- Signal_Analysis/ — analysis routines:
  - SignalAnalysis.py — DFT/ANM/CS implementations, frequency extraction and error metrics.
- utils/ — miscellaneous helpers:
  - NoiseVisualization.py — functions to visualize noise models in quantum hardware.



## Setup
Run these commands to create and activate a virtual environment, then install packages from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To access IBM Quantum Processors, create secret.json with following content
{
    "TOKEN": "your_token",
    "INSTANCE":"your_instance"
}

To generate a token and an instance sign in on https://quantum.cloud.ibm.com/ 