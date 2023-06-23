from ..nodes import FunctionNode
from ..typefunctions import (
    check_input_square,
    copy_from_input_to_output,
    check_input_dimension,
    check_inputs_multiplicable_mat,
)

from numpy import multiply


class CovmatrixFromCormatrix(FunctionNode):
    """Compute covariance matrix from correlation matrix:
    Vₖₘ=Cₖₘσₖσₘ
    """

    _mode: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "C→V")

        self._add_pair("matrix", "matrix", output_kws={"positional": True})
        self._add_input("sigma", positional=False)

    def _fcn(self):
        self.inputs.touch()
        C = self.inputs["matrix"].data
        sigma = self.inputs["sigma"].data

        V = self.outputs["matrix"].data

        multiply(C, sigma[None, :], out=V)
        multiply(V, sigma[:, None], out=V)

    def _typefunc(self) -> None:
        check_input_square(self, "matrix")
        check_input_dimension(self, "sigma", 1)
        check_inputs_multiplicable_mat(self, "matrix", "sigma")
        copy_from_input_to_output(self, slice(None), slice(None))
