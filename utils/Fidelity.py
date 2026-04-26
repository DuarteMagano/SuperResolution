import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def get_operator_from_circuit(qc, transpiled=False, skip_ops=("barrier", "measure", "delay")):
    """
    Return the unitary matrix of a circuit.

    If transpiled=True, rebuild the circuit using only active qubits,
    ordered by original logical-qubit order when layout information is available.
    """
    if not transpiled:
        clean_qc = qc.remove_final_measurements(inplace=False)
        return Operator(clean_qc).data
    
    # Collect qubits touched by non-skipped operations.
    used = {
        q
        for inst in qc.data
        if inst.operation.name not in skip_ops
        for q in inst.qubits
    }

    def logical_order_key(q):
        # Default fallback: physical qubit index.
        physical_index = qc.find_bit(q).index

        # Use transpiler layout, if available, to recover original logical order.
        if qc.layout is None or qc.layout.initial_layout is None:
            return physical_index

        physical_to_virtual = qc.layout.initial_layout.get_physical_bits()
        virtual_q = physical_to_virtual.get(physical_index)

        if virtual_q is None:
            return physical_index

        if virtual_q in qc.layout.input_qubit_mapping:
            return qc.layout.input_qubit_mapping[virtual_q]

        return physical_index

    active = sorted(used, key=logical_order_key)

    qmap = {q: i for i, q in enumerate(active)}

    small_qc = QuantumCircuit(len(active))
    small_qc.global_phase = qc.global_phase

    for inst in qc.data:
        if inst.operation.name in skip_ops:
            continue

        small_qc.append(
            inst.operation.copy(),
            [small_qc.qubits[qmap[q]] for q in inst.qubits],
        )

    return Operator(small_qc).data

def get_fidelity_operators(U, V):
    return np.abs(np.trace(U.conj().T @ V)) ** 2 / np.size(U)