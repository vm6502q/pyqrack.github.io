# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import math
import ctypes
from .qrack_system import Qrack
from .pauli import Pauli


class QrackSimulator:
    """Interface for all the QRack functionality.

    Attributes:
        qubitCount(int): Number of qubits that are to be simulated.
        cloneSid: TODO
        isSchmidtDecomposeMulti: TODO
        isSchmidtDecompose: TODO
        isStabilizerHybrid: TODO
        isBinaryDecisionTree: TODO
        is1QbFusion: TODO
        isPaged: TODO
        isCpuGpuHybrid:TODO
        isOpenCL: TODO
        isHostPointer: TODO
        pyzxCircuit: TODO
    """

    def __init__(
        self,
        qubitCount=-1,
        cloneSid=-1,
        isSchmidtDecomposeMulti=True,
        isSchmidtDecompose=True,
        isStabilizerHybrid=True,
        isBinaryDecisionTree=False,
        is1QbFusion=False,
        isPaged=True,
        isCpuGpuHybrid=True,
        isOpenCL=True,
        isHostPointer=False,
        pyzxCircuit=None,
    ):
        pass

    # standard gates

    ## single-qubits gates
    def x(self, q):
        """Applies X gate.

        Applies the Pauli “X” operator to the qubit at position “q.”
        The Pauli “X” operator is equivalent to a logical “NOT.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def y(self, q):
        """Applies Y gate.

        Applies the Pauli “Y” operator to the qubit at “q.”
        The Pauli “Y” operator is equivalent to a logical “NOT" with
        permutation phase.

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def z(self, q):
        """Applies Z gate.

        Applies the Pauli “Z” operator to the qubit at “q.”
        The Pauli “Z” operator flips the phase of `|1>`

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def h(self, q):
        """Applies H gate.

        Applies the Hadarmard operator to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def s(self, q):
        """Applies S gate.

        Applies the 1/4 phase rotation to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def t(self, q):
        """Applies T gate.

        Applies the 1/8 phase rotation to the qubit at “q.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def adjs(self, q):
        """Adjoint of S gate

        Applies the gate equivalent to the inverse of S gate.

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def adjt(self, q):
        """Adjoint of T gate

        Applies the gate equivalent to the inverse of T gate.

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def u(self, q, th, ph, la):
        """General unitary gate.

        Applies a gate guaranteed to be unitary.
        Spans all possible single bit unitary gates.

        `U(theta, phi, lambda) = RZ(phi + pi/2)RX(theta)RZ(lambda - pi/2)`

        Args:
            q: the qubit number on which the gate is applied to.
            th: theta
            ph: phi
            la: lambda

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mtrx(self, m, q):
        """Operation from matrix.

        Applies arbitrary operation defined by the given matrix.

        Args:
            m: row-major complex matrix which defines the operator.
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def r(self, b, ph, q):
        """Rotation gate.

        Rotate the qubit along the given pauli basis by the given angle.


        Args:
            b: Pauli basis
            ph: rotation angle
            q: the qubit number on which the gate is applied to

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def exp(self, b, ph, q):
        """Arbitrary exponentiation

        `exp(b, theta) = e^{i*theta*[b_0 . b_1 ...]}`
        where `.` is the tensor product.


        Args:
            b: Pauli basis
            ph: coefficient of exponentiation
            q: the qubit number on which the gate is applied to

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    ## multi-qubit gates
    def mcx(self, c, q):
        """Multi-controlled X gate

        If all controlled qubits are `|1>` then the target qubit is flipped.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcy(self, c, q):
        """Multi-controlled Y gate

        If all controlled qubits are `|1>` then the Pauli "Y" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcz(self, c, q):
        """Multi-controlled Z gate

        If all controlled qubits are `|1>` then the Pauli "Z" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mch(self, c, q):
        """Multi-controlled H gate

        If all controlled qubits are `|1>` then the Hadarmard gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcs(self, c, q):
        """Multi-controlled S gate

        If all controlled qubits are `|1>` then the "S" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mct(self, c, q):
        """Multi-controlled T gate

        If all controlled qubits are `|1>` then the "T" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcadjs(self, c, q):
        """Multi-controlled adjs gate

        If all controlled qubits are `|1>` then the adjs gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcadjt(self, c, q):
        """Multi-controlled adjt gate

        If all controlled qubits are `|1>` then the adjt gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcu(self, c, q, th, ph, la):
        """Multi-controlled arbitraty unitary

        If all controlled qubits are `|1>` then the unitary gate described by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.
            th: theta
            ph: phi
            la: lambda

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcmtrx(self, c, m, q):
        """Multi-controlled arbitraty operator

        If all controlled qubits are `|1>` then the arbitrary operation by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits
            m: row-major complex matrix which defines the operator
            q: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def macx(self, c, q):
        """Anti multi-controlled X gate

        If all controlled qubits are `|0>` then the target qubit is flipped.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def macy(self, c, q):
        """Anti multi-controlled Y gate

        If all controlled qubits are `|0>` then the Pauli "Y" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def macz(self, c, q):
        """Anti multi-controlled Z gate

        If all controlled qubits are `|0>` then the Pauli "Z" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mach(self, c, q):
        """Anti multi-controlled H gate

        If all controlled qubits are `|0>` then the Hadarmard gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def macs(self, c, q):
        """Anti multi-controlled S gate

        If all controlled qubits are `|0>` then the "S" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mact(self, c, q):
        """Anti multi-controlled T gate

        If all controlled qubits are `|0>` then the "T" gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def macadjs(self, c, q):
        """Anti multi-controlled adjs gate

        If all controlled qubits are `|0>` then the adjs gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def macadjt(self, c, q):
        """Anti multi-controlled adjt gate

        If all controlled qubits are `|0>` then the adjt gate is applied to
        the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def macu(self, c, q, th, ph, la):
        """Anti multi-controlled arbitraty unitary

        If all controlled qubits are `|0>` then the unitary gate described by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits.
            q: target qubit.
            th: theta
            ph: phi
            la: lambda

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def macmtrx(self, c, m, q):
        """Anti multi-controlled arbitraty operator

        If all controlled qubits are `|0>` then the arbitrary operation by
        parameters is applied to the target qubit.

        Args:
            c: list of controlled qubits.
            m: row-major complex matrix which defines the operator.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def multiplex1_mtrx(self, c, q, m):
        """Multiplex gate

        A multiplex gate with a single target and an arbitrary number of
        controls.

        Args:
            c: list of controlled qubits.
            m: row-major complex matrix which defines the operator.
            q: target qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mx(self, q):
        """Multi X-gate

        Applies the Pauli “X” operator on all qubits.

        Args:
            q: list of qubits to apply X on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def my(self, q):
        """Multi Y-gate

        Applies the Pauli “Y” operator on all qubits.

        Args:
            q: list of qubits to apply Y on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mz(self, q):
        """Multi Z-gate

        Applies the Pauli “Z” operator on all qubits.

        Args:
            q: list of qubits to apply Z on.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcr(self, b, ph, c, q):
        """Multi-controlled arbitrary rotation.

        If all controlled qubits are `|1>` then the arbitrary rotation by
        parameters is applied to the target qubit.
        Applies the Pauli “Z” operator on all qubits.

        Args:
            b: row-major complex matrix which defines the operator.
            ph: coefficient of exponentiation.
            c: list of controlled qubits.
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcexp(self, b, ph, cs, q):
        """Multi-controlled arbitrary exponentiation

        If all controlled qubits are `|1>` then the the target qubit is
        exponentiated an pauli basis basis with coefficient.

        Args:
            b: Pauli basis
            ph: coefficient of exponentiation.
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def swap(self, qi1, qi2):
        """Swap Gate

        Swaps the qubits at two given positions.

        Args:
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def iswap(self, qi1, qi2):
        """Swap Gate with phase.

        Swaps the qubits at two given positions.
        If the bits are different then there is additional phase of `i`.

        Args:
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def adjiswap(self, qi1, qi2):
        """Swap Gate with phase.

        Swaps the qubits at two given positions.
        If the bits are different then there is additional phase of `-i`.

        Args:
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def fsim(self, th, ph, qi1, qi2):
        """Fsim gate.

        The 2-qubit “fSim” gate
        Useful in the simulation of particles with fermionic statistics

        Args:
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def cswap(self, c, qi1, qi2):
        """Controlled-swap Gate

        Swaps the qubits at two given positions if the control qubits are `|1>`

        Args:
            c: list of controlled qubits.
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def acswap(self, c, qi1, qi2):
        """Anti controlled-swap Gate

        Swaps the qubits at two given positions if the control qubits are `|0>`

        Args:
            c: list of controlled qubits.
            qi1: First position of qubit.
            qi2: Second position of qubit.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    # standard operations
    def m(self, q):
        """Measurement gate

        Measures the qubit at "q" and returns Boolean value.
        This operator is not unitary & is probabilistic in nature.

        Args:
            q: qubit to measure

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Measurement result.
        """
        pass

    def force_m(self, q, r):
        """Force-Measurement gate

        Acts as if the measurement is applied and the result obtained is `r`

        Args:
            q: qubit to measure
            r: the required result

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Measurement result.
        """
        pass

    def m_all(self):
        """Measure-all gate

        Measures measures all qubits.
        This operator is not unitary & is probabilistic in nature.

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Measurement result of all qubits.
        """
        pass

    def measure_pauli(self, b, q):
        """Pauli Measurement gate

        Measures the qubit at "q" with the given pauli basis.
        This operator is not unitary & is probabilistic in nature.

        Args:
            b: Pauli basis
            q: qubit to measure

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            Measurement result.
        """
        pass

    def measure_shots(self, q, s):
        """Multi-shot measurement operator

        Measures the qubit at "q" with the given pauli basis.
        This operator is not unitary & is probabilistic in nature.

        Args:
            q: qubit to measure
            s: number of shots

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            list of measurement result.
        """
        pass

    def reset_all(self):
        """Reset gate

        Resets all gates to `|0>`

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    # arithmetic-logic-unit (ALU)
    def _split_longs(self, a):
        """Split operation

        Splits the given integer into 64 bit numbers.


        Args:
            a: number to split

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            list of split numbers.
        """
        pass

    def _split_longs_2(self, a, m):
        """Split simultanoues operation

        Splits 2 integers into same number of 64 bit numbers.

        Args:
            a: first number to split
            m: second number to split

        Raises:
            RuntimeError: QrackSimulator raised an exception.

        Returns:
            pair of lists of split numbers.
        """
        pass

    def add(self, a, q):
        """Add integer to qubit

        Adds the given integer to the given set of qubits.
        Note that the integer is first split into 64 bit numbers.

        Args:
            a: first number to split
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def sub(self, a, q):
        """Subtract integer to qubit

        Subtracts the given integer to the given set of qubits.
        Note that the integer is first split into 64 bit numbers.

        Args:
            a: first number to split
            q: list of qubits to subtract the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def adds(self, a, s, q):
        """Add integer to qubit

        TODO
        Adds the given integer to the given set of qubits with sign.
        Note that the integer is first split into 64 bit numbers.

        Args:
            a: number to add
            s:
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """

    def subs(self, a, s, q):
        """Subtract integer to qubit

        TODO
        Subtracts the given integer to the given set of qubits with sign.
        Note that the integer is first split into 64 bit numbers.

        Args:
            a: number to subtract
            s:
            q: list of qubits to subtract the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mul(self, a, q, o):
        """Multiplies integer to qubit

        TODO
        Multiplies the given integer to the given set of qubits.
        Note that the integer is first split into 64 bit numbers.

        TODO
        Args:
            a: number to multiply
            q: list of qubits to multiply the number
            o:

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def div(self, a, q, o):
        """Divides qubit by integer

        TODO
        Divides the given qubits to the given set of integer.
        Note that the integer is first split into 64 bit numbers.

        TODO
        Args:
            a: number to divide by
            q: qubits to divide
            o:

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def muln(self, a, m, q, o):
        pass

    def divn(self, a, m, q, o):
        pass

    def pown(self, a, m, q, o):
        pass

    def mcadd(self, a, c, q):
        """Controlled-add

        Adds the given integer to the given set of qubits if all controlled
        qubits are `|1>`.
        Note that the integer is first split into 64 bit numbers.

        Args:
            a: number to add.
            c: list of controlled qubits.
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcsub(self, a, c, q):
        """Controlled-subtract

        Subtracts the given integer to the given set of qubits if all controlled
        qubits are `|1>`.
        Note that the integer is first split into 64 bit numbers.

        Args:
            a: number to subtract.
            c: list of controlled qubits.
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcmul(self, a, c, q, o):
        """Controlled-multiply

        Multiplies the given integer to the given set of qubits if all controlled
        qubits are `|1>`.
        Note that the integer is first split into 64 bit numbers.

        Args:
            a: number to multiply
            c: list of controlled qubits.
            q: list of qubits to add the number

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcdiv(self, a, c, q, o):
        """Controlled-divide.

        Divides the given qubits by the given number if all controlled
        qubits are `|1>`.
        Note that the integer is first split into 64 bit numbers.

        Args:
            a: number to divide by
            c: list of controlled qubits.
            q: qubits to divide
            o:

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def mcmuln(self, a, c, m, q, o):
        pass

    def mcdivn(self, a, c, m, q, o):
        pass

    def mcpown(self, a, c, m, q, o):
        pass

    def lda(self, qi, qv, t):
        pass

    def adc(self, s, qi, qv, t):
        pass

    def sbc(self, s, qi, qv, t):
        pass

    def hash(self, q, t):
        pass

    # boolean logic gates
    def qand(self, qi1, qi2, qo):
        """Logical AND

        Logical AND of 2 qubits whose result is stored in the target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def qor(self, qi1, qi2, qo):
        """Logical OR

        Logical OR of 2 qubits whose result is stored in the target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def qxor(self, qi1, qi2, qo):
        """Logical XOR

        Logical exlusive-OR of 2 qubits whose result is stored in the target
        qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def qnand(self, qi1, qi2, qo):
        """Logical NAND

        Logical NAND of 2 qubits whose result is stored in the target
        qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def qnor(self, qi1, qi2, qo):
        """Logical NOR

        Logical NOR of 2 qubits whose result is stored in the target
        qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def qxnor(self, qi1, qi2, qo):
        """Logical XOR

        Logical exlusive-NOR of 2 qubits whose result is stored in the target
        qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def cland(self, ci, qi, qo):
        """Classical AND

        Logical AND with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def clor(self, ci, qi, qo):
        """Classical OR

        Logical OR with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def clxor(self, ci, qi, qo):
        """Classical XOR

        Logical exlusive-OR with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def clnand(self, ci, qi, qo):
        """Classical NAND

        Logical NAND with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def clnor(self, ci, qi, qo):
        """Classical NOR

        Logical NOR with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass

    def clxnor(self, ci, qi, qo):
        """Classical XNOR

        Logical exlusive-NOR with one qubit and one classical bit whose result is
        stored in target qubit.

        Args:
            qi1: qubit 1
            qi2: qubit 2
            qo: target qubit

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
        pass
