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

@njit(void(float64[:], float64[:], float64[:], float64, float64, float64))
def _ibdxsecO1(enu, costheta, result, me, mp, mn):
    print("here", enu[0])
