from numpy import sqrt
from scipy.linalg import cholesky

from ...core.input_handler import MissingInputAddPair
from ...core.node import Node
from ...core.type_functions import check_has_inputs, check_input_matrix_or_diag, copy_from_input_to_output


class Cholesky(Node):
    """Compute the Cholesky decomposition of a matrix V=LL̃ᵀ
    1d input is considered to be a diagonal of square matrix"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler",
            MissingInputAddPair(input_fmt="matrix", output_fmt="L"),
        )
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "V→L")

        self._functions_dict.update({"square": self._fcn_square, "diagonal": self._fcn_diagonal})

    def _fcn_square(self):
        """Compute Cholesky decomposition using `scipy.linalg.cholesky`
        NOTE: inplace computation (`overwrite_a=True`) works only for
        the F-based arrays. As soon as by default C-arrays are used,
        transposition produces an F-array (view). Transposition with
        `lower=False` produces a lower matrix in the end.
        """
        self.inputs.touch()

        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _output[:] = _input
            cholesky(_output.T, overwrite_a=True, lower=False)  # produces L (!) inplace

    def _fcn_diagonal(self):
        """Compute "Cholesky" decomposition using of a diagonal of a square matrix.
        Elementwise sqrt is used.
        """
        self.inputs.touch()

        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            sqrt(_input, out=_output)

    def _typefunc(self) -> None:
        check_has_inputs(self)
        ndim = check_input_matrix_or_diag(self, slice(None), check_square=True)
        copy_from_input_to_output(self, slice(None), slice(None))

        if ndim == 2:
            self.function = self._functions_dict["square"]
            self.labels.mark = "V→L"
        else:
            self.function = self._functions_dict["diagonal"]
            self.labels.mark = "sqrt(Vᵢ)"
