from numpy.typing import ArrayLike
from numpy import sin
from numba import njit, void, float64

from dagflow.typefunctions import check_input_dtype
from dagflow.input_extra import MissingInputAddPair
from dagflow.nodes import FunctionNode
from dagflow.input import Input
from dagflow.output import Output

class IBDXsecO1(FunctionNode):
    """Inverse beta decay cross section by Vogel and Beacom"""
    __slots__ = ('_enu', '_ctheta', '_result')

    _enu: Input
    _ctheta: Input
    _result: Output

    def __init__(self, name, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddPair())
        super().__init__(name, *args, **kwargs)

        self._enu = self.add_input('enu', positional=True, keyword=True)
        self._ctheta = self.add_input('costheta', positional=True, keyword=True)
        self._result = self.add_output('result', positional=True, keyword=True)

        self._me = self.add_input('ElectronMass', positional=False, keyword=True)
        self._mp = self.add_input('ProtonMass', positional=False, keyword=True)
        self._mn = self.add_input('NeutronMass', positional=False, keyword=True)

    def _fcn(self, _, inputs, outputs):
        _ibdxsecO1(
                self._enu.data.ravel(),
                self._ctheta.data.ravel(),
                self._result.data.ravel(),
                self._me.data[0],
                self._mp.data[0],
                self._mn.data[0]
                )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_input_dtype(self, slice(None), 'd')

        for inp, out in zip(self.inputs, self.outputs):
            out.dd.axes_edges = inp.dd.axes_edges
            out.dd.axes_nodes = inp.dd.axes_nodes
            out.dd.dtype = inp.dd.dtype
            out.dd.shape = inp.dd.shape

# NOTE: these functions are used only in non-numba case
from numpy.typing import NDArray
from numpy import double
from numpy import sqrt, power as pow, pi
@njit(void(float64[:], float64[:], float64[:], float64, float64, float64), cache=True)
def _ibdxsecO1(
    EnuIn: NDArray[double],
    CosThetaIn: NDArray[double],
    Result: NDArray[double],
    ElectronMass: float,
    ProtonMass: float,
    NeutronMass: float
):
    ElectronMass2 = pow(ElectronMass, 2)
    NeutronMass2 = pow(NeutronMass, 2)
    # ProtonMass2 = pow(ProtonMass, 2)
    NucleonMass = 0.5*(NeutronMass + ProtonMass)
    EnuThreshold = 0.5 * (NeutronMass2 / (ProtonMass - ElectronMass) - ProtonMass + ElectronMass)

    DeltaNP = NeutronMass-ProtonMass
    y2 = 0.5*(pow(DeltaNP, 2)-ElectronMass2)

    # TODO: pass as input
    PhaseFactor = 1.7152
    g = 1.2601
    f = 1.
    f2 = 3.706
    gsq = g*g
    fsq = f*f
    # f2sq = f2*f2

    Qe=1.
    Hbar=1.
    NeutronLifeTime=1.

    for i, (Enu, ctheta) in enumerate(zip(EnuIn, CosThetaIn)):
        if Enu<EnuThreshold:
            continue

        Ee0 = Enu - DeltaNP
        if Ee0<=ElectronMass:
            Result[i]=0.0
            continue

        pe0 = sqrt(Ee0*Ee0 - ElectronMass2)
        ve0 = pe0 / Ee0
        ElectronMass5 = ElectronMass2 * ElectronMass2 * ElectronMass
        sigma0 = 2.* pi * pi / (PhaseFactor*(fsq+3.*gsq)*ElectronMass5*NeutronLifeTime/(1.E-6*Hbar/Qe))

        Ee1 = Ee0 * ( 1.0 - Enu/NucleonMass * ( 1.0 - ve0*ctheta ) ) - y2/NucleonMass
        if Ee1 <= ElectronMass:
            Result[i]=0.0
            continue
        pe1 = sqrt(Ee1*Ee1 - ElectronMass2)
        ve1 = pe1/Ee1

        sigma1a = sigma0*0.5 * ( ( fsq + 3.*gsq ) + ( fsq - gsq ) * ve1 * ctheta ) * Ee1 * pe1

        gamma_1 = 2.0 * g * ( f + f2 ) * ( ( 2.0 * Ee0 + DeltaNP ) * ( 1.0 - ve0 * ctheta ) - ElectronMass2/Ee0 )
        gamma_2 = ( fsq + gsq ) * ( DeltaNP * ( 1.0 + ve0*ctheta ) + ElectronMass2/Ee0 )
        A = ( ( Ee0 + DeltaNP ) * ( 1.0 - ctheta/ve0 ) - DeltaNP )
        gamma_3 = ( fsq + 3.0*gsq )*A
        gamma_4 = ( fsq -     gsq )*A*ve0*ctheta

        sigma1b = -0.5 * sigma0 * Ee0 * pe0 * ( gamma_1 + gamma_2 + gamma_3 + gamma_4 ) / NucleonMass

        Result[i]=sigma1a + sigma1b

