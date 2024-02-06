from typing import TYPE_CHECKING

from numpy import multiply

from ..nodes import FunctionNode
from ..typefunctions import check_input_dimension
from ..typefunctions import check_input_square
from ..typefunctions import check_inputs_multiplicable_mat
from ..typefunctions import copy_from_input_to_output

if TYPE_CHECKING:
    from ..input import Input
    from ..output import Output


class CovmatrixFromCormatrix(FunctionNode):
    """Compute covariance matrix from correlation matrix:
    Vₖₘ=Cₖₘσₖσₘ
    """

    __slots__ = ("_mode", "_sigma", "_cormatrix", "_covmatrix")

    _mode: str
    _sigma: "Input"
    _cormatrix: "Input"
    _covmatrix: "Output"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, allowed_kw_inputs=("sigma", "cormatrix"))
        self._labels.setdefault("mark", "C→V")

        self._cormatrix, self._covmatrix = self._add_pair(
            "cormatrix", "covmatrix", output_kws={"positional": True}
        )
        self._sigma = self._add_input("sigma", positional=False)

    def _fcn(self):
        self.inputs.touch()
        C = self._cormatrix.data
        sigma = self._sigma.data
        V = self._covmatrix.data

        multiply(C, sigma[None, :], out=V)
        multiply(V, sigma[:, None], out=V)

    def _typefunc(self) -> None:
        check_input_square(self, "cormatrix")
        check_input_dimension(self, "sigma", 1)
        check_inputs_multiplicable_mat(self, "cormatrix", "sigma")
        copy_from_input_to_output(self, slice(None), slice(None))
