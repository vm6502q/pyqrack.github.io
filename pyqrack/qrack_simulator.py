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
    def x(self, q: int):
        """Applies X gate.

        Applies the Pauli “X” operator to the qubit at “qubitIndex.”
        The Pauli “X” operator is equivalent to a logical “NOT.”

        Args:
            q: the qubit number on which the gate is applied to.

        Raises:
            RuntimeError: QrackSimulator raised an exception.
        """
