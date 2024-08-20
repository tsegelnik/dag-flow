from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import TYPE_CHECKING, Sequence

from ..metanode import MetaNode
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
    )

    _stat_cov: Cache
    _dict_jacobian: dict[str, list[Jacobian]]
    _dict_cov_pars: dict[str, list[Node | Output]]
    _dict_cov_syst_part: dict[str, list[Node]]
    _dict_cov_syst: dict[str, Node]
    _dict_cov_full: dict[str, Node]

    _cov_sum_syst: Node | None
    _cov_sum_full: Node | None

    def __init__(self, *, labels: Mapping = {}):
        super().__init__()

        self._dict_jacobian = defaultdict(list)
        self._dict_cov_syst_part = defaultdict(list)
        self._dict_cov_syst = {}
        self._dict_cov_full = {}

        self._cov_sum_syst = None
        self._cov_sum_full = None

        self._init_stat("stat_cov", label=labels.get("stat_cot", {}))

    def _init_stat(self, name: str, *, label={}):
        self._stat_cov = Cache(name, label=label)
        self._stat_cov()
        self._add_node(self._stat_cov, kw_inputs=("input",), merge_inputs=("input",))
        self.inputs.make_positional("input")

    def compute_covariance_for(
        self,
        name: str,
        parameter_groups: Sequence,
        *,
        parameter_covariance_matrices: Sequence | None = None,
        label={},
    ) -> Node:
        if name in self._dict_jacobian:
            raise RuntimeError(f"Covariance group {name} already defined")

        jacobians = self._dict_jacobian[name]
        matrices = self._dict_cov_syst_part[name]
        for i, pars in enumerate(parameter_groups):
            jacobian = Jacobian(f"Jacobian: {name}", parameters=pars)
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
                    f"V syst: {name} ({i})", left=jacobian, square=pars_covmat
                )
            else:
                vsyst_part = MatrixProductDDt.from_args(f"V syst: {name} ({i})", matrix=jacobian)
            matrices.append(vsyst_part)

        if len(matrices) > 1:
            vsyst = Sum.from_args(f"V syst: {name}", *matrices)
            for matrix in matrices:
                self._add_node(matrix)
            self._add_node(vsyst)
        else:
            vsyst = matrices[0]
            self._add_node(vsyst)
        self._dict_cov_syst[name] = vsyst

        vfull = SumMatOrDiag.from_args(f"V total: {name}", self._stat_cov, vsyst)
        self._add_node(vfull, outputs_pos=False)
        self._dict_cov_full[name] = vfull

        return vfull

    def compute_covariance_sum(
        self,
        name: str,
        *,
        label={},
    ) -> Node:
        if self._cov_sum_syst is not None:
            raise RuntimeError(f"Sum of covariance matrices already computed")

        vsyst_part = list(self._dict_cov_syst.values())
        if len(vsyst_part) > 1:
            self._cov_sum_syst = Sum.from_args(f"V syst sum: {name}", *vsyst_part)
            self._add_node(self._cov_sum_syst)
        else:
            self._cov_sum_syst = vsyst_part[0]

        self._cov_sum_full = SumMatOrDiag.from_args(
            f"V total: {name}", self._stat_cov, self._cov_sum_syst
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
