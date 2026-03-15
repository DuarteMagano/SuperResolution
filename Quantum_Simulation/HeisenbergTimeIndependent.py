from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp, Operator
import os
from tqdm import tqdm
from itertools import chain
from scipy.linalg import expm
#from qiskit.tools.monitor import job_monitor
from dataclasses import asdict
from collections import Counter

class SpinModel():
    def __init__(self, noise = False, real_hardware = False, service = None, simulator = AerSimulator()):
        self.noise = noise
        self.real_hardware = real_hardware
        self.service = service
        self.simulator = simulator

        if self.noise:
            #self.backend = service.backend('ibm_brisbane')
            self.backend = service.least_busy(operational=True, simulator=False)
            self.backend = service.backend("ibm_torino")
            noise_model = NoiseModel.from_backend(self.backend)
            #self.basis_gates = noise_model.basis_gates
            #self.coupling_map = self.backend.configuration().coupling_map
            self.simulator = AerSimulator(noise_model = noise_model)
            self.sampler = sampler = Sampler(self.backend, options={"default_shots": 10000})
            self.sampler.options.dynamical_decoupling.enable = True
            self.sampler.options.twirling.enable_gates = True


    def get_prob0_job(self, job):
        bit_array = job.result()[0].data["c"]
        raw = bit_array.array  
        num_bits = bit_array.num_bits
        bits_flat = np.unpackbits(raw, axis=1, bitorder='little')
        bitstrings = [''.join(str(bit) for bit in row[:num_bits]) for row in bits_flat]
        counts = Counter(bitstrings)
        return counts.get('0',0) / sum(counts.values())

    
    def XX(self, theta):
        XX = QuantumCircuit(2)
        XX.cx(0, 1)
        XX.rx(theta, 0)
        XX.cx(0, 1)
        return XX.to_gate(label="XX")

    def YY(self, theta):
        YY = QuantumCircuit(2)
        YY.sdg(1)
        YY.cx(0, 1)
        YY.ry(theta, 0)
        YY.cx(0, 1)
        YY.s(1)
        return YY.to_gate(label="YY")

    def ZZ(self, theta):
        ZZ = QuantumCircuit(2)
        ZZ.cx(0, 1)
        ZZ.rz(theta, 1)
        ZZ.cx(0, 1)
        return ZZ.to_gate(label="ZZ")
    
    def CU(self, t, n):
        theta = 2 * t / n
        qc = QuantumCircuit(3)
        for step in range(0, n):
            for qubits in ([0, 1], [1, 2]):
                qc.append(self.XX(theta), qubits)
                qc.append(self.YY(theta), qubits)
                qc.append(self.ZZ(theta), qubits)

        return qc.to_gate().control()
    
    def S2_CU(self, t, n):
        theta = t / (2 * n)
        qc = QuantumCircuit(3)
        qubits_list = [[0, 1], [1, 2]]
        qubits_list= list(chain(qubits_list, qubits_list[::-1]))
        gates = [self.XX(theta), self.YY(theta), self.ZZ(theta)]
        gates = list(chain(gates, gates[::-1]))

        for step in range(0, n):
            for qubits in qubits_list:
                for gate in gates:
                    qc.append(gate, qubits)

        return qc.to_gate().control()
    
    def S4_CU(self, t, n):
        u2 = 1 / (4 - 4 ** (1 / 3) )
        S2_1 = self.S2_CU(u2 * t, n)
        S2_2 =  self.S2_CU((1 - 4 * u2) * t, n)

        qc = QuantumCircuit(4)
        qc.append(S2_1, [0, 1, 2, 3])
        qc.append(S2_1, [0, 1, 2, 3])
        qc.append(S2_2, [0, 1, 2, 3])
        qc.append(S2_1, [0, 1, 2, 3])
        qc.append(S2_1, [0, 1, 2, 3])

        return qc.to_gate()
    

    def get_probability_0(self, t, coeff, order_formula, n = 30, shots = 10000):
        if order_formula == "analytical":
            Zero = Statevector.from_label("0")
            One = Statevector.from_label("1")
            qc = QuantumCircuit(4)
            qc.h(0)
            qc.append(UnitaryGate(SpinModel().U_heis3(t)).control(), [0,1,2,3])
            qc.barrier()
            qc.h(0)
            Q = Operator(qc).data
            psi = np.dot(Q, self.mapping(coeff) ^ Zero)
            return np.sum(np.abs(psi[::2] ** 2))

        #cr = ClassicalRegister(1, name="cr")
        qc = QuantumCircuit(4, 1)
        qc.initialize(coeff, [1, 2, 3])

        qc.h(0)
        if order_formula == 1: 
            qc.append(self.CU(t, n), [0, 1, 2, 3])
        if order_formula == 2:
             qc.append(self.S2_CU(t, n), [0, 1, 2, 3])
        if order_formula == 4:
             qc.append(self.S4_CU(t, n), [0, 1, 2, 3])
        qc.barrier()

        qc.h(0)
        qc.barrier()

        #qc.measure(0, 0)
        if self.real_hardware:
            qc_decomposed = qc.decompose(reps = 3)
            qc_decomposed.measure(0, 0)
            qc_transpiled = transpile(qc_decomposed, backend= self.backend, optimization_level = 0)

            #sampler = Sampler(mode=self.backend)
            job = self.sampler.run([qc_transpiled], shots=1000)
            print("Running on real quantum hardware — job ID:", job.job_id())
            return self.get_prob0_job(job)

        elif self.noise:
            qc_decomposed = qc.decompose(reps = 3)
            qc_decomposed.measure(0, 0) 
            qc_transpiled = transpile(qc_decomposed, backend= self.backend, optimization_level = 0)
            job = self.simulator.run(qc_transpiled, shots = 1000)
            noisy_result = job.result()
            counts = noisy_result.get_counts()

        else:
            qc.measure(0, 0)
            qc_transpiled = transpile(qc, self.simulator)
            result = self.simulator.run(qc_transpiled, shots = shots).result()
            counts = result.get_counts()

        total_shots = sum(counts.values())
        prob_dist = {state: count / total_shots for state, count in counts.items()} 
        return prob_dist.get('0', 0)
    
    

    def Analysis(self, coeff, t_max = 10 * np.pi, N = 200, order_formula = 1, n = 50):
        t = - t_max
        dt =  2 * t_max / N
        data = [[], []]

        #print(filename_prefix)
        filename_prefix = "Time Independent/"
        filename_prefix += f"Order_{order_formula}/"
        if self.noise:
            filename_prefix += "noisy_"
        if self.real_hardware:
            filename_prefix += "hardware_"

        #filename_prefix = filename_prefix #+ f"order_{order_formula}_"
        filename_time =  f"{filename_prefix}n_{n}_t_max_{np.round(t_max/np.pi)}_pi_time_{'_'.join(map(str, coeff))}.txt"
        filename_energy =  f"{filename_prefix}t_max_{np.round(t_max/np.pi)}_pi_energy_{'_'.join(map(str, coeff))}.txt"

        with open(filename_time, 'w') as f:
            for i in range(N + 1):
                data[0].append(t)
                data[1].append(self.get_probability_0(t, coeff, order_formula, n))            
                #print(data[0][-1], data[1][-1])
                f.write(f"{data[0][-1]} {data[1][-1]}\n")
                t += dt      

        #DFT
        signal_fft = np.abs(np.fft.fft(np.array(data[1]) - 1 / 2, norm = "ortho")) 
        energies = np.fft.fftfreq(N + 1, dt) * 2 * np.pi
        with open(filename_energy, 'w') as f:
            for i in range(len(energies)):
                f.write(f"{energies[i]} {signal_fft[i]}\n") 


    def H_heis3(self):
        I = Pauli('I')
        X = Pauli('X')
        Y = Pauli('Y')
        Z = Pauli('Z')
        XXs = SparsePauliOp(["IXX", "XXI"], coeffs=[1, 1])
        YYs = SparsePauliOp(["IYY", "YYI"], coeffs=[1, 1])
        ZZs = SparsePauliOp(["IZZ", "ZZI"], coeffs=[1, 1])
        # Sum interactions
        H = XXs + YYs + ZZs
        # Return Hamiltonian
        return H
    
    def U_heis3(self, t):
        H = self.H_heis3()
        H_matrix = self.H_heis3().to_matrix()
        U = expm(-1j * t * H_matrix)
        return U
    
    def mapping(self, coeff, string_format = False):
        index = np.argmax(coeff)  
        n = int(np.log2(len(coeff))) 
        if string_format:
            return format(index, f'0{n}b')[::-1]
        return Statevector.from_label(format(index, f'0{n}b')[::-1])  