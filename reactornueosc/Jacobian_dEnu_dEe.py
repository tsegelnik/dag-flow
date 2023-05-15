from numba import njit, void, float64

from dagflow.input_extra import MissingInputAddPair
from dagflow.nodes import FunctionNode
from dagflow.input import Input
from dagflow.output import Output

from typing import Mapping

class Jacobian_dEnu_dEe(FunctionNode):
    """Enu(Ee, cosθ)"""
    __slots__ = (
        '_enu', '_ctheta',
        '_result',
        '_const_me', '_const_mp', '_const_mn',
    )

    _enu: Input
    _ee: Input
    _ctheta: Input
    _result: Output

    _const_me: Input
    _const_mp: Input
    _const_mn: Input

    def __init__(self, name, *args, label: Mapping={}, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddPair())
        label = {
                'text': r'Energy conversion Jacobian dEν/dEdep',
                'latex': r'$dE_{\nu}/dE_{\rm dep}$',
                'axis': r'$dE_{\nu}/dE_{\rm dep}$',
                }
        label.update(label)
        super().__init__(name, *args, label=label, **kwargs)

        self._enu = self.add_input('enu', positional=True, keyword=True)
        self._ee = self.add_input('ee', positional=True, keyword=True)
        self._ctheta = self.add_input('costheta', positional=True, keyword=True)
        self._result = self.add_output('result', positional=True, keyword=True)

        self._const_me   = self.add_input('ElectronMass', positional=False, keyword=True)
        self._const_mp   = self.add_input('ProtonMass', positional=False, keyword=True)
        self._const_mn   = self.add_input('NeutronMass', positional=False, keyword=True)

    def _fcn(self, _, inputs, outputs):
        _jacobian_dEnu_dEe(
            self._enu.data.ravel(),
            self._ee.data.ravel(),
            self._ctheta.data.ravel(),
            self._result.data.ravel(),
            self._const_me.data[0],
            self._const_mp.data[0],
            self._const_mn.data[0]
        )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from dagflow.typefunctions import (
            assign_output_axes_from_inputs,
            check_input_dimension,
            check_inputs_equivalence,
            copy_from_input_to_output
        )
        check_input_dimension(self, slice(0, 3), 2)
        check_inputs_equivalence(self, slice(0, 3))
        copy_from_input_to_output(self, 'ee', 'result', edges=False, nodes=False)
        assign_output_axes_from_inputs(self, ('ee', 'costheta'), 'result', assign_nodes=True)


# NOTE: these functions are used only in non-numba case
from numpy.typing import NDArray
from numpy import double
from numpy import sqrt, power as pow, square
@njit(void(float64[:], float64[:], float64[:], float64[:],
           float64, float64, float64), cache=True)
def _jacobian_dEnu_dEe(
    EnuIn: NDArray[double],
    EeIn: NDArray[double],
    CosThetaIn: NDArray[double],
    Result: NDArray[double],
    ElectronMass: float,
    ProtonMass: float,
    NeutronMass: float
):
    ElectronMass2 = pow(ElectronMass, 2)
    NeutronMass2 = pow(NeutronMass, 2)
    ProtonMass2 = pow(ProtonMass, 2)

    delta = 0.5*(NeutronMass2-ProtonMass2-ElectronMass2)/ProtonMass

    for i, (Enu, Ee, ctheta) in enumerate(zip(EnuIn, EeIn, CosThetaIn)):
        Ve2 = 1.0 - ElectronMass2 / (Ee*Ee)
        if Ve2<=0:
            Result[i] = 0.0
            continue

        Ve = sqrt(Ve2)
        nominator = ProtonMass + Enu*(1.0-ctheta/Ve)
        denominator = ProtonMass - Ee*(1-Ve*ctheta)
        if denominator>0:
            Result[i] = nominator/denominator
        else:
            Result[i] = 0.0
