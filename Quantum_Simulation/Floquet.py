from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp, Operator
import os
from tqdm import tqdm
from itertools import chain
from scipy.linalg import expm
#from qiskit.tools.monitor import job_monitor
from dataclasses import asdict

class Floquet_Simulation():
    def __init__(self, T, parameters, dimension=1, backend = None, noisy = False, real_hardware = False):
        self.T = T
        self.w = 2 * np.pi / T
        self.parameters = parameters
        self.dimension = dimension
        self.backend = backend
        self.noisy = noisy
        self.real_hardware = real_hardware
        noise_model = None
        if self.real_hardware:
            pass # Code to be completed for real hardware execution
        elif self.noisy:
            noise_model = NoiseModel.from_backend(backend)
            self.runner = AerSimulator(noise_model=noise_model)
        else:
            self.runner = AerSimulator()

    def Simpson_integral(self, function, t_in, t_f):
        return 1 / 6 * (t_f - t_in) * (function(t_in) + 4 * function((t_in + t_f) / 2) + function(t_f))

    
    def U_circular_polarized_I_order(self, dt, t_k):
        qc = QuantumCircuit(1)
        h2_0 = self.parameters[1] * np.cos(self.w * t_k)
        h2_1 = self.parameters[1] * np.sin(self.w * t_k)

        qc.rz(2 * self.parameters[0] * dt, [0])
        qc.rx(2 * h2_0 * dt, [0])
        qc.ry(2 * h2_1 * dt, [0])

        return qc.to_gate()




    def U_circular_polarized_II_order(self, dt, t_k):
        qc = QuantumCircuit(1)
        h2_0 = self.parameters[1] * np.cos(self.w * t_k)
        h2_1 = self.parameters[1] * np.sin(self.w * t_k)

        qc.rz(self.parameters[0] * dt, [0])
        qc.rx(h2_0 * dt, [0])
        qc.ry(2 * h2_1 * dt, [0])
        qc.rx(h2_0 * dt, 0)
        qc.rz(self.parameters[0] * dt, [0])
        return qc.to_gate()
    

    def U_linear_polarized_II_order(self, dt, t_k):
        qc = QuantumCircuit(1)
        h2 = self.parameters[1] * np.cos(self.w * t_k) 

        qc.rz(self.parameters[0] * dt, [0])
        qc.rx(2 * h2 * dt, [0])  
        qc.rz(self.parameters[0] * dt, [0])
        return qc.to_gate()
    
    def U_linear_polarized_IV_order(self, dt, t_in, t_f):
        qc = QuantumCircuit(1)
        s = 1 / (2 - 2 ** (1 / 3))

        h1, h2 = self.parameters[0], self.parameters[1]
        omega = self.w
        delta = t_f - t_in

        alpha_1 = h1 * delta
        alpha_2 = h2 * integrate.quad(np.cos, t_in, t_f)[0]
        mycos = lambda t : np.cos(self.w * t)
        func = lambda t_1: self.parameters[0] * self.parameters[1] * (integrate.quad(mycos, t_in, t_1)[0] - (t_1 - t_in) * mycos(t_1))
        if alpha_2 == 0:
            alpha_12 = 0
        else:
            alpha_12 = integrate.quad(func, t_in, t_f)[0] / 2
        #alpha_2 = (h2 / omega) * (np.sin(omega * t_f) - np.sin(omega * t_in))
        #alpha_12 = (h1 * h2) / (2 * omega ** 2) * (2 - 2 * np.cos(omega * delta) - omega * delta * np.sin(omega * delta))

        u = alpha_12 / alpha_2

        qc.rz(s * alpha_1 / 2 - u, [0])
        qc.rx(s * alpha_2, [0])
        qc.rz((1 - s) / 2 * alpha_1, [0])
        qc.rx((1 - 2 * s) * alpha_2, [0])
        qc.rz((1 - s) / 2 * alpha_1, [0])
        qc.rx(s * alpha_2, [0])
        qc.rz(s * alpha_1 / 2 + u, [0])

        return qc.to_gate()

    
    def evolution_operator(self, t_f, t_0, num_Trotter_steps, Polarization, Order_formula):
        qc = QuantumCircuit(self.dimension)
        dt = (t_f - t_0) / num_Trotter_steps

        if Polarization == "Circular":
            for i in range(num_Trotter_steps):
                t_k = t_0 + (i + 1 / 2) * dt
                if Order_formula == "First":
                    qc.append(self.U_circular_polarized_I_order(dt, t_k), [0])
                elif Order_formula == "Second":
                    qc.append(self.U_circular_polarized_II_order(dt, t_k), [0])     ################ dt/2?

        elif Polarization == "Linear":
            if Order_formula == "Second":
                for i in range(num_Trotter_steps):
                    t_k = t_0 + (i + 1 / 2) * dt
                    qc.append(self.U_circular_polarized_II_order(dt, t_k), [0])         

            elif Order_formula == "Fourth":
                t_i = 0
                t = dt
                for i in range(num_Trotter_steps):
                    qc.append(self.U_linear_polarized_IV_order(dt, t_i, t), [0])
                    t_i += dt
                    t += dt
            else:
                raise ValueError("Order must be either Second or Fourth")

        else:
            raise ValueError("Polarization must be either Circular or Linear")
        return qc.to_gate()
    
    def SWAP_test(self, coeff, U):
        qc = QuantumCircuit(self.dimension + 1, 1)
        qc.initialize(coeff, range(self.dimension))
        qc.h(0)
        qc.append(U.control(), range(self.dimension + 1))
        qc.h(0)

        qc.barrier()
        qc.measure(0, 0)
        return qc
    
    def get_distribution0(self, coeff=[1, 0], Polarization="Circular", Order_formula="Second", num_Trotter_steps =100, num_points=3000, shots =10000, optimization_level=0):
        TIME = []; Prob = []
        prefix = "noisy_" if self.noisy else ""
        prefix = "quantum_hardware_" if self.real_hardware else prefix
        fname = f"Files/QuantumSimulation_{prefix}{Polarization}_Polarization_T={self.T}_Order={Order_formula}_t_MAX={int(num_points / 2) * self.T}_Trotter_steps={num_Trotter_steps}.txt"
        with open(fname, "w") as f:
            for t in self.T * np.arange(-int(num_points / 2), int(num_points / 2) + 1, 1):
                U = self.evolution_operator(t, 0, num_Trotter_steps, Polarization, Order_formula)                  
                
                qc = self.SWAP_test(coeff, U)
                qc = transpile(qc, backend=self.backend, optimization_level=optimization_level) if self.backend else transpile(qc, self.runner, optimization_level=optimization_level)
                job = self.runner.run(qc, shots=shots)
                Prob = job.result().get_counts().get('0', 0) / shots             
                f.write(f"{t} {Prob} \n")
    
    def quasi_energy_circular(self, Polarization = "Circular"):
        h1, h2 = self.parameters[0], self.parameters[1]
        if Polarization == "Linear":
            h2 /= 2
        energy = self.w / 2 - np.sqrt((h1- self.w / 2) ** 2 + h2 ** 2)
        if energy < self.w / 2:
            return energy
        return energy - self.w / 2
    
    def circular_analytical_distribution0(self, coeff=[1, 0], num_points=3000, write=False, plot=True):
        w_exp = self.quasi_energy_circular()
        TIME = self.T * np.arange(-int(num_points / 2), int(num_points / 2) + 1, 1)
        Prob = (1 + np.cos(w_exp * TIME)) / 2
        if plot:
            plt.plot(TIME, Prob)
            plt.show()
        if write:
            fname = f"Files/QuantumSimulation_Circular_Polarization_T={self.T}_Order=Analytical_t_MAX={int(num_points / 2) * self.T}.txt"
            np.savetxt(fname, TIME, Prob)

    