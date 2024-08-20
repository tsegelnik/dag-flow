from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import TYPE_CHECKING, Sequence

from multikeydict.nestedmkdict import NestedMKDict

from ..metanode import MetaNode
from ..parameters import GaussianParameter, NormalizedGaussianParameter
from . import Sum
from .Cache import Cache
from .Jacobian import Jacobian
from .MatrixProductDDt import MatrixProductDDt
from .MatrixProductDVDt import MatrixProductDVDt
from .SumMatOrDiag import SumMatOrDiag

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..node import Node, Output


class CovarianceMatrixGroup(MetaNode):
    __slots__ = (
        "_stat_cov",
        "_dict_jacobian",
        "_dict_cov_pars",
        "_dict_cov_syst_part",
        "_dict_cov_syst",
        "_dict_cov_full",
        "_cov_sum_syst",
        "_cov_sum_full",
        "_parameters",
        "_ignore_duplicated_paramters",
    )

    _stat_cov: Cache
    _dict_jacobian: dict[str, list[Jacobian]]
    _dict_cov_pars: dict[str, list[Node | Output]]
    _dict_cov_syst_part: dict[str, list[Node]]
    _dict_cov_syst: dict[str, Node]
    _dict_cov_full: dict[str, Node]

    _cov_sum_syst: Node | None
    _cov_sum_full: Node | None

    _parameters: set[GaussianParameter | NormalizedGaussianParameter]
    _ignore_duplicated_paramters: bool

    def __init__(self, *, labels: Mapping = {}, ignore_duplicated_parameters: bool = False):
        super().__init__()

        self._dict_jacobian = defaultdict(list)
        self._dict_cov_syst_part = defaultdict(list)
        self._dict_cov_syst = {}
        self._dict_cov_full = {}

        self._cov_sum_syst = None
        self._cov_sum_full = None

        self._parameters = set()
        self._ignore_duplicated_paramters = ignore_duplicated_parameters

        self._init_stat("stat_cov", label=labels.get("stat_cot", {}))

    def _init_stat(self, name: str, *, label={}):
        self._stat_cov = Cache(name, label=label)
        self._stat_cov()
        self._add_node(self._stat_cov, kw_inputs=("input",), merge_inputs=("input",))
        self.inputs.make_positional("input")

    def get_parameters_count(self) -> int:
        return len(self._parameters)

    def add_covariance_for(
        self,
        name: str,
        parameter_groups: (
            NormalizedGaussianParameter
            | GaussianParameter
            | Sequence[NormalizedGaussianParameter]
            | Sequence[GaussianParameter]
            | Sequence[Sequence[NormalizedGaussianParameter]]
            | Sequence[Sequence[GaussianParameter]]
            | Sequence[NestedMKDict]
            | NestedMKDict
        ),
        *,
        parameter_covariance_matrices: Sequence | None = None,
        label={},
    ) -> Node:
        if name in self._dict_jacobian:
            raise RuntimeError(f"Covariance group {name} already defined")

        jacobians = self._dict_jacobian[name]
        matrices = self._dict_cov_syst_part[name]
        parameter_groups_clean = self._get_parameter_groups(parameter_groups)
        npars_total = 0
        for i, pars in enumerate(parameter_groups_clean):
            self._check_pars_unique(pars)
            npars = len(pars)
            npars_total += npars

            jacobian = Jacobian(f"Jacobian ({npars}): {name}", parameters=pars)
            jacobian()
            self._add_node(jacobian, kw_inputs=("input",), merge_inputs=("input",))
            self.inputs.make_positional("input", index=0)
            jacobians.append(jacobian)

            pars_covmat = None
            if parameter_covariance_matrices is not None:
                with suppress(IndexError):
                    pars_covmat = parameter_covariance_matrices[i]

            if pars_covmat:
                vsyst_part = MatrixProductDVDt.from_args(
                    f"V syst ({npars}): {name} ({i})", left=jacobian, square=pars_covmat
                )
            else:
                vsyst_part = MatrixProductDDt.from_args(
                    f"V syst ({npars}): {name} ({i})", matrix=jacobian
                )
            matrices.append(vsyst_part)

        if len(matrices) > 1:
            vsyst = Sum.from_args(f"V syst ({npars_total}): {name}", *matrices)
            for matrix in matrices:
                self._add_node(matrix)
            self._add_node(vsyst)
        else:
            vsyst = matrices[0]
            self._add_node(vsyst)
        self._dict_cov_syst[name] = vsyst

        vfull = SumMatOrDiag.from_args(f"V total ({npars_total}): {name}", self._stat_cov, vsyst)
        self._add_node(vfull, outputs_pos=False)
        self._dict_cov_full[name] = vfull

        return vfull

    def add_covariance_sum(
        self,
        name: str = "sum",
        *,
        label={},
    ) -> Node:
        if self._cov_sum_syst is not None:
            raise RuntimeError(f"Sum of covariance matrices already computed")

        npars = len(self._parameters)

        vsyst_part = list(self._dict_cov_syst.values())
        if len(vsyst_part) > 1:
            self._cov_sum_syst = Sum.from_args(f"V syst sum ({npars}): {name}", *vsyst_part)
            self._add_node(self._cov_sum_syst)
        else:
            self._cov_sum_syst = vsyst_part[0]

        self._cov_sum_full = SumMatOrDiag.from_args(
            f"V total ({npars}): {name}", self._stat_cov, self._cov_sum_syst
        )
        self._add_node(self._cov_sum_full, outputs_pos=True)

        return self._cov_sum_full

    def compute_jacobians(self):
        for jacobians in self._dict_jacobian.values():
            for jacobian in jacobians:
                jacobian.compute()

    def update_matrices(self):
        self._stat_cov.recache()
        self.compute_jacobians()

        for cov_full in self._dict_cov_full.values():
            cov_full.touch()

        if self._cov_sum_full:
            self._cov_sum_full.touch()

    def _get_parameter_groups(
        self,
        parameter_groups: (
            NormalizedGaussianParameter
            | GaussianParameter
            | Sequence[NormalizedGaussianParameter]
            | Sequence[GaussianParameter]
            | Sequence[Sequence[NormalizedGaussianParameter]]
            | Sequence[Sequence[GaussianParameter]]
            | Sequence[NestedMKDict]
            | NestedMKDict
        ),
    ) -> Sequence[Sequence[NormalizedGaussianParameter]] | Sequence[Sequence[GaussianParameter]]:
        match parameter_groups:
            case NormalizedGaussianParameter() | GaussianParameter():
                return ((parameter_groups,),)  # pyright: ignore [reportReturnType]
            case [NormalizedGaussianParameter() | GaussianParameter(), *_]:
                return (parameter_groups,)  # pyright: ignore [reportReturnType]
            case NestedMKDict():
                return (tuple(parameter_groups.walkvalues()),)  # pyright: ignore [reportReturnType]
            case [[NormalizedGaussianParameter() | GaussianParameter(), *_], *_]:
                return parameter_groups
            case [NestedMKDict(), *_]:
                return tuple(
                    tuple(group.walkvalues()) for group in parameter_groups
                )  # pyright: ignore [reportReturnType]

        raise RuntimeError("Invalid parameter_groups type")

    def _check_pars_unique(
        self, pars: Sequence[NormalizedGaussianParameter] | Sequence[GaussianParameter]
    ):
        for par in pars:
            if not self._ignore_duplicated_paramters and par in self._parameters:
                break

            self._parameters.add(par)
        else:
            return

        raise RuntimeError(f"CovarianceMatrixGroup: duplicated parameter {par!s}")
