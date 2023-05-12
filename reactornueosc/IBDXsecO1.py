from dagflow.input_extra import MissingInputAddPair
from dagflow.nodes import FunctionNode
from dagflow.input import Input
from dagflow.output import Output

class IBDXsecO1(FunctionNode):
    """Inverse beta decay cross section by Vogel and Beacom"""
    __slots__ = (
        '_enu', '_ctheta',
        '_result',
        '_const_me', '_const_mp', '_const_mn',
        '_const_taun',
        '_const_fps', '_const_g', '_const_f', '_const_f2',
    )

    _enu: Input
    _ctheta: Input
    _result: Output

    _const_me: Input
    _const_mp: Input
    _const_mn: Input
    _const_taun: Input
    _const_fps: Input
    _const_g: Input
    _const_f: Input
    _const_f2: Input

    def __init__(self, name, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddPair())
        super().__init__(name, *args, **kwargs)

        self._enu = self.add_input('enu', positional=True, keyword=True)
        self._ctheta = self.add_input('costheta', positional=True, keyword=True)
        self._result = self.add_output('result', positional=True, keyword=True)

        self._const_me   = self.add_input('ElectronMass', positional=False, keyword=True)
        self._const_mp   = self.add_input('ProtonMass', positional=False, keyword=True)
        self._const_mn   = self.add_input('NeutronMass', positional=False, keyword=True)
        self._const_taun = self.add_input('NeutronLifeTime', positional=False, keyword=True)
        self._const_fps  = self.add_input('PhaseSpaceFactor', positional=False, keyword=True)
        self._const_g    = self.add_input('g', positional=False, keyword=True)
        self._const_f    = self.add_input('f', positional=False, keyword=True)
        self._const_f2   = self.add_input('f2', positional=False, keyword=True)

    def _fcn(self, _, inputs, outputs):
        _ibdxsecO1(
                self._enu.data.ravel(),
                self._ctheta.data.ravel(),
                self._result.data.ravel(),
                self._const_me.data[0], self._const_mp.data[0], self._const_mn.data[0], self._const_taun.data[0],
                self._const_fps.data[0], self._const_g.data[0], self._const_f.data[0], self._const_f2.data[0],
                )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from dagflow.typefunctions import (
            check_input_dtype,
            check_input_dimension,
            check_inputs_equivalence,
            copy_from_input_to_output,
            assign_output_axes_from_inputs
        )

        check_input_dtype(self, slice(None), 'd')
        check_input_dimension(self, slice(0, 1), 2)
        check_inputs_equivalence(self, slice(0, 1))
        copy_from_input_to_output(self, 'enu', 'result', edges=False, nodes=False)
        assign_output_axes_from_inputs(self, ('enu', 'costheta'), 'result', assign_nodes=True)

from scipy.constants import value as constant
_constant_hbar = constant('reduced Planck constant')
_constant_qe = constant('elementary charge')
from numba import njit, void, float64, float32
from numpy.typing import NDArray
from numpy import double, pi, sqrt, power as pow
@njit([
    void(float64[:], float64[:], float64[:],
         float64, float64, float64, float64,
         float64, float64, float64, float64),
    void(float64[:], float64[:], float64[:],
         float32, float32, float32, float32,
         float32, float32, float32, float32),
    ],
    cache=True)
def _ibdxsecO1(
    EnuIn: NDArray[double], CosThetaIn: NDArray[double], Result: NDArray[double],
    ElectronMass: float, ProtonMass: float, NeutronMass: float, NeutronLifeTime: float,
    const_fps: float, const_g: float, const_f: float, const_f2: float
):
    ElectronMass2 = pow(ElectronMass, 2)
    NeutronMass2 = pow(NeutronMass, 2)
    NucleonMass = 0.5*(NeutronMass + ProtonMass)
    EnuThreshold = 0.5 * (NeutronMass2 / (ProtonMass - ElectronMass) - ProtonMass + ElectronMass)

    DeltaNP = NeutronMass-ProtonMass
    const_y2 = 0.5*(pow(DeltaNP, 2)-ElectronMass2)

    const_gsq = const_g*const_g
    const_fsq = const_f*const_f

    sigma0_constant = 1.e6*_constant_qe/_constant_hbar
    ElectronMass5 = ElectronMass2 * ElectronMass2 * ElectronMass
    sigma0 = (2.* pi * pi) / (const_fps*(const_fsq+3.*const_gsq)*ElectronMass5*NeutronLifeTime*sigma0_constant)

    for i, (Enu, ctheta) in enumerate(zip(EnuIn, CosThetaIn)):
        if Enu<EnuThreshold:
            Result[i]=0.0
            continue

        Ee0 = Enu - DeltaNP
        if Ee0<=ElectronMass:
            Result[i]=0.0
            continue

        pe0 = sqrt(Ee0*Ee0 - ElectronMass2)
        ve0 = pe0 / Ee0

        Ee1 = Ee0 * ( 1.0 - Enu/NucleonMass * ( 1.0 - ve0*ctheta ) ) - const_y2/NucleonMass
        if Ee1 <= ElectronMass:
            Result[i]=0.0
            continue
        pe1 = sqrt(Ee1*Ee1 - ElectronMass2)
        ve1 = pe1/Ee1

        sigma1a = sigma0*0.5 * ( ( const_fsq + 3.*const_gsq ) + ( const_fsq - const_gsq ) * ve1 * ctheta ) * Ee1 * pe1

        gamma_1 = 2.0 * const_g * ( const_f + const_f2 ) * ( ( 2.0 * Ee0 + DeltaNP ) * ( 1.0 - ve0 * ctheta ) - ElectronMass2/Ee0 )
        gamma_2 = ( const_fsq + const_gsq ) * ( DeltaNP * ( 1.0 + ve0*ctheta ) + ElectronMass2/Ee0 )
        A = ( ( Ee0 + DeltaNP ) * ( 1.0 - ctheta/ve0 ) - DeltaNP )
        gamma_3 = ( const_fsq + 3.0*const_gsq )*A
        gamma_4 = ( const_fsq -     const_gsq )*A*ve0*ctheta

        sigma1b = -0.5 * sigma0 * Ee0 * pe0 * ( gamma_1 + gamma_2 + gamma_3 + gamma_4 ) / NucleonMass

        Result[i]=sigma1a + sigma1b

