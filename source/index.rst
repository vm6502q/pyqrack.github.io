PyQrack
=================

Introduction
------------
`PyQrack <https://github.com/vm6502q/pyqrack>`_ is a pure Python `ctypes` wrapper on the C++11 `Qrack <https://github.com/vm6502q/qrack>`_ library, for quantum computing simulation.

An introductory talk to Qrack can be found here `Intro to Qrack: a framework for fast quantum simulation by Daniel Strano | Quantum Software Talks <https://www.youtube.com/watch?v=yxyqJDC4SUo>`_.

Hardware Compilation
---------------

Efficient Unitary Clifford+RZ Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`QStabilizerHybrid` is classically efficient for the gate set "Clifford+RZ," (or "Clifford+T,") except for measurement, since v1.14. The entire unitary portion of circuit simulation, before measurement, has a polynomial-complexity simulation algorithm, in space and time requirements. If measuring across the full width of the simulator, or sampling, measurement (alone) scales exponentially in space requirements proportional to (less than or up to) the number of non-Clifford RZ (or T) gates, and exponentially in time requirements proportional to base logical qubit count in the simulator instance.

No special considerations are necessary to engage this simulation mode: simply restrict your gate set to Clifford+RZ, when using any simulator that properly includes the `QStabilizerHybrid` layer, such as the default optimal simulator stack.

Output Unitary Clifford+RZ Simulation For Quantum Hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is theoretically possible to use the Clifford+RZ improvements of v1.14 to compile for hardware. (`QUnit` "Schmidt decomposition cannot be used over `QStabilizerHybrid`, for this.)

Since v1.15, it is now possible to output `QStabilizerHybrid` state to file, (not while using `QUnit`). This is done with instance method `out_to_file(filename)`, on a simulator instance. With class method `QrackSimulator.file_to_qiskit_circuit(filename)`, we perform the algorithm that follows, to produce a Qiskit circuit. The files have the following format, by line:

[Logical qubit count]

[Stabilizer qubit count, including ancillae]

[Stabilizer x/z/r generators, one row per line, "tableau" format, repeated for logical qubit count of rows x2]

[Per-qubit MPS buffers, 2x2 complex matrices, row-major order, one matrix per line, repeated for stabilizer qubit count of rows]


For example:

3

3

1 1 0 0 1 0 2

0 1 0 1 0 0 0

0 0 0 0 0 1 0

0 0 0 1 0 1 2

0 0 0 1 1 0 0

1 1 1 0 1 0 0

(1,0) (0,0) (0,0) (1,0)

(1,0) (0,0) (0,0) (1,0)

(0,0) (0.707107,-0.707107) (0,1) (0,0)

is a valid file, with 0 ancillae. It is theoretically relatively easy to prepare this result of unitary circuit simulation on a quantum hardware device: first prepare the stabilizer state, (with purely Clifford gates,) then apply the (potentially non-Clifford) 2x2 matrices over the same sequential qubit index order. This can represent a universal quantum state of the logical qubits.

It might be helpful to look at Qiskit's `Clifford <https://qiskit.org/documentation/stubs/qiskit.quantum_info.Clifford.html>_` and `Clifford.to_circuit() <https://qiskit.org/documentation/stubs/qiskit.quantum_info.Clifford.to_circuit.html#qiskit.quantum_info.Clifford.to_circuit>_` to convert this stabilizer state representation to a state preparation circuit.

.. toctree::
    :maxdepth: 2
    :hidden:

    Introduction <self>
