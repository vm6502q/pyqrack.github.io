# (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import ctypes

from .qrack_system import Qrack

_IS_QISKIT_AVAILABLE = True
try:
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.compiler.transpiler import transpile
    import numpy as np
    import math
except ImportError:
    _IS_QISKIT_AVAILABLE = False

class QrackCircuit:
    """Class that exposes the QCircuit class of Qrack

    QrackCircuit allows the user to specify a unitary circuit, before running it.
    Upon running the state, the result is a QrackSimulator state. Currently,
    measurement is not supported, but measurement can be run on the resultant
    QrackSimulator.

    Attributes:
        cid(int): Qrack ID of this circuit
    """

    def __init__(self, clone_cid = -1):
        if clone_cid < 0:
            self.cid = Qrack.qrack_lib.init_qcircuit()
        else:
            self.cid = Qrack.qrack_lib.init_qcircuit_clone(clone_cid)

    def __del__(self):
        if self.cid is not None:
            Qrack.qrack_lib.destroy_qcircuit(self.cid)
            self.cid = None

    def _ulonglong_byref(self, a):
        return (ctypes.c_ulonglong * len(a))(*a)

    def _double_byref(self, a):
        return (ctypes.c_double * len(a))(*a)

    def _complex_byref(self, a):
        t = [(c.real, c.imag) for c in a]
        return self._double_byref([float(item) for sublist in t for item in sublist])

    def get_qubit_count(self):
        """Get count of qubits in circuit

        Raises:
            RuntimeError: QracQrackCircuitNeuron C++ library raised an exception.
        """
        return Qrack.qrack_lib.get_qcircuit_qubit_count(self.cid)

    def swap(self, q1, q2):
        """Add a 'Swap' gate to the circuit

        Args:
            q1: qubit index #1
            q2: qubit index #2

        Raises:
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
        Qrack.qrack_lib.qcircuit_swap(self.cid, q1, q2)

    def mtrx(self, m, q):
        """Operation from matrix.

        Applies arbitrary operation defined by the given matrix.

        Args:
            m: row-major complex list representing the operator.
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackCircuit C++ library raised an exception.
        """
        Qrack.qrack_lib.qcircuit_append_1qb(self.cid, self._complex_byref(m), q)

    def ucmtrx(self, c, m, q, p):
        """Multi-controlled single-target-qubit gate

        Specify a controlled gate by its control qubits, its single-qubit
        matrix "payload," the target qubit, and the permutation of qubits
        that activates the gate.

        Args:
            c: list of controlled qubits
            m: row-major complex list representing the operator.
            q: target qubit
            p: permutation of target qubits

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """

        p_list = [((p >> i) & 1) for i in range(len(c))]
        p_list = [x for _, x in sorted(zip(c, p_list))]
        p = 0
        for i in range(len(p_list)):
            p |= p_list[i] << i

        Qrack.qrack_lib.qcircuit_append_mc(
            self.cid, self._complex_byref(m), len(c), self._ulonglong_byref(c), q, p
        )

    def run(self, qsim):
        """Run circuit on simulator

        Run the encoded circuit on a specific simulator. The
        result will remain in this simulator.

        Args:
            qsim: QrackSimulator on which to run circuit

        Raises:
            RuntimeError: QrackCircuit raised an exception.
        """
        Qrack.qrack_lib.qcircuit_run(self.cid, qsim.sid)
        qsim._throw_if_error()

        qb_count = self.get_qubit_count()
        if qsim._qubitCount < qb_count:
            qsim._qubitCount = qb_count

    def out_to_file(self, filename):
        """Output optimized circuit to file

        Outputs the (optimized) circuit to a file named
        according to the "filename" parameter.

        Args:
            filename: Name of file
        """
        Qrack.qrack_lib.qcircuit_out_to_file(self.cid, filename.encode('utf-8'))

    def in_from_file(self, filename):
        """Read in optimized circuit from file

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter.

        Args:
            filename: Name of file
        """
        Qrack.qrack_lib.qcircuit_in_from_file(self.cid, filename.encode('utf-8'))

    def file_to_qiskit_circut(filename):
        """Convert an output file to a Qiskit circuit

        Reads in an (optimized) circuit from a file named
        according to the "filename" parameter and outputs
        a Qiskit circuit.

        Args:
            filename: Name of file

        Raises:
            RuntimeErorr: Before trying to file_to_qiskit_circuit() with
                QrackCircuit, you must install Qiski, numpy, and math!
        """
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        tokens = []
        with open(filename, 'r') as file:
            tokens = file.read().split()

        i = 0
        num_qubits = int(tokens[i])
        i = i + 1
        circ = QuantumCircuit(num_qubits, num_qubits)

        num_gates = int(tokens[i])
        i = i + 1

        for g in range(num_gates):
            target = int(tokens[i])
            i = i + 1

            control_count = int(tokens[i])
            i = i + 1
            controls = []
            for j in range(control_count):
                controls.append(int(tokens[i]))
                i = i + 1

            payload_count = int(tokens[i])
            i = i + 1
            payloads = {}
            for j in range(payload_count):
                key = int(tokens[i])
                i = i + 1
                op = np.zeros((2,2), dtype=complex)
                row = []
                for _ in range(2):
                    amp = tokens[i].replace("(","").replace(")","").split(',')
                    row.append(float(amp[0]) + float(amp[1])*1j)
                    i = i + 1
                op[0][0] = row[0]
                op[0][1] = row[1]

                row = []
                for _ in range(2):
                    amp = tokens[i].replace("(","").replace(")","").split(',')
                    row.append(float(amp[0]) + float(amp[1])*1j)
                    i = i + 1
                op[1][0] = row[0]
                op[1][1] = row[1]

                # Qiskit has a lower tolerance for deviation from numerically unitary.

                th = math.acos(np.real(op[0][0])) * 2
                s = math.sin(th / 2)
                if s < 1e-6:
                    ph = np.real(np.log(op[1][1] / math.cos(th / 2)) / 1j) / 2
                    lm = ph
                else:
                    ph = np.real(np.log(op[1][0] / s) / 1j)
                    lm = np.real(-np.log(op[0][1] / s) / 1j)

                c = math.cos(th / 2)
                s = math.sin(th / 2)
                op3 = np.exp(1j * (ph + lm)) * c
                if np.abs(op[1][1] - op3) > 1e6:
                    print("Warning: gate ", str(g), ", payload ", str(j), " might not be unitary!")

                op[0][0] = c
                op[0][1] = -np.exp(1j * lm) * s
                op[1][0] = np.exp(1j * ph) * s
                op[1][1] = op3

                payloads[key] = np.array(op)

            gate_list = []
            for j in range(1 << control_count):
                if j in payloads:
                    gate_list.append(payloads[j])
                else:
                    gate_list.append(np.array([[1, 0],[0, 1]]))

            circ.uc(gate_list, controls, target)

        return circ

    def in_from_qiskit_circuit(self, circ):
        """Read a Qiskit circuit into a QrackCircuit

        Reads in a circuit from a Qiskit `QuantumCircuit`

        Args:
            circ: Qiskit circuit

        Raises:
            RuntimeErorr: Before trying to file_to_qiskit_circuit() with
                QrackCircuit, you must install Qiski, numpy, and math!
        """
        if not _IS_QISKIT_AVAILABLE:
            raise RuntimeError(
                "Before trying to file_to_qiskit_circuit() with QrackCircuit, you must install Qiskit, numpy, and math!"
            )

        basis_gates = ["u", "cx"]
        circ = transpile(circ, basis_gates=basis_gates, optimization_level=3)
        for gate in circ.data:
            o = gate.operation
            if o.name == "u":
                th = o.params[0]
                ph = o.params[1]
                lm = o.params[2]

                c = math.cos(th / 2)
                s = math.sin(th / 2)

                op = []
                op.append(c)
                op.append(-np.exp(1j * lm) * s)
                op.append(np.exp(1j * ph) * s)
                op.append(np.exp(1j * (ph + lm)) * c)
                self.mtrx(op, gate.qubits[0].index)
            else:
                ctrls = []
                for c in gate.qubits[1:]:
                    ctrls.append(c.index)
                self.ucmtrx(ctrls, [0, 1, 1, 0], gate.qubits[0].index, 1)
