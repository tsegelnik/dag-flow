#!/usr/bin/env python

from numpy import allclose, arange, array, diag, finfo
from numpy.linalg import cholesky
from pytest import mark, raises

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Product, Sum
from dagflow.lib.Array import Array
from dagflow.lib.CovarianceMatrixGroup import CovarianceMatrixGroup
from dagflow.parameters import Parameters


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("correlated", (False, True))
def test_CovarianceMatrixGroup(dtype, correlated: bool, testname):
    """
    Test CovarianceMatrixGroup on
    y = a*a*a*x + b*b*x + c*x + d
    """
    size = 10

    value = [2.1, -2.3, 3.2, 1.1]
    x = arange(size, dtype=dtype)
    sigma = array([0.1, 0.5, 2.0, 3.0])
    if correlated:
        correlations = [
            [1.0, 0.5, -0.5, 0.1],
            [0.5, 1.0, 0.0, 0.0],
            [-0.5, 0.0, 1.0, 0.0],
            [0.1, 0.0, 0.0, 1.0],
        ]

        vpar = (sigma[:, None] @ sigma[None, :]) * correlations
        lpar = cholesky(vpar)
    else:
        vpar = diag(sigma**2)
        lpar = diag(sigma).astype(dtype)
        correlations = None

    with Graph(close_on_exit=True) as graph:
        X = Array("x", x)
        pars = Parameters.from_numbers(
            value,
            names=list("abcd"),
            sigma=sigma,
            dtype=dtype,
            correlation=correlations,
            provide_covariance=True,
        )
        A, B, C, D = pars.outputs()

        first = Product.from_args("a³x", A, A, A, X)
        second = Product.from_args("b²x", B, B, X)
        third = Product.from_args("cx", C, X)
        Y = Sum.from_args("f(x)=a³x+b²x+cx+d", first, second, third, D)

        cm = CovarianceMatrixGroup()
        cm.compute_covariance_for("covmat AB", [pars.norm_parameters[:2]])
        cm.compute_covariance_for("covmat CD", [pars.norm_parameters[2:]])

        cm2 = CovarianceMatrixGroup()
        cm2.compute_covariance_for(
            "covmat ABCD",
            [pars.parameters],
            parameter_covariance_matrices=[pars.constraint._covariance_node],
        )

        cm3 = CovarianceMatrixGroup()
        cm3.compute_covariance_for("covmat ABCD", [pars.norm_parameters])

        if correlated:
            cm.compute_covariance_sum("covmat ABCD")

            with raises(RuntimeError):
                cm.compute_covariance_sum("covmat ABCD")
        else:
            cm.compute_covariance_for(
                "covmat A,B", [pars.norm_parameters[:1], pars.norm_parameters[1:2]]
            )
            cm.compute_covariance_sum("covmat ABABCD")

            with raises(RuntimeError):
                cm.compute_covariance_sum("covmat ABABCD")

        with raises(RuntimeError):
            cm.compute_covariance_for("covmat AB", [pars.norm_parameters[:2]])

        Y >> cm
        Y >> cm2
        Y >> cm3

    if not correlated:
        jac_A = cm._dict_jacobian["covmat A,B"][0].get_data().T
        jac_B = cm._dict_jacobian["covmat A,B"][1].get_data().T
    jac_AB = cm._dict_jacobian["covmat AB"][0].get_data()
    jac_CD = cm._dict_jacobian["covmat CD"][0].get_data()

    jac_ABCD_2 = cm2._dict_jacobian["covmat ABCD"][0].get_data()
    jac_ABCD_3 = cm3._dict_jacobian["covmat ABCD"][0].get_data()

    if not correlated:
        vsyst_AcB = cm._dict_cov_syst["covmat A,B"].get_data()
    vsyst_AB = cm._dict_cov_syst["covmat AB"].get_data()
    vsyst_CD = cm._dict_cov_syst["covmat CD"].get_data()
    vsyst = cm._cov_sum_syst.get_data()
    vsyst2 = cm2._dict_cov_syst["covmat ABCD"].get_data()
    vsyst3 = cm3._dict_cov_syst["covmat ABCD"].get_data()

    vfull_AB = cm._dict_cov_full["covmat AB"].get_data()
    vfull_CD = cm._dict_cov_full["covmat CD"].get_data()
    vfull = cm._cov_sum_full.get_data()
    vfull2 = cm2._dict_cov_full["covmat ABCD"].get_data()
    vfull3 = cm3._dict_cov_full["covmat ABCD"].get_data()

    jac_check0 = array(
        [
            3.0 * value[0] ** 2 * x,
            2.0 * value[1] ** 1 * x,
            1.0 * value[2] ** 0 * x,
            0.0 * value[3] ** 0 * x + 1.0,
        ],
    ).T

    vstat = diag(Y.get_data())

    jac_check = jac_check0 @ lpar

    jac_check_A = jac_check[:, 0]
    jac_check_B = jac_check[:, 1]
    jac_check_AB = jac_check[:, :2]
    jac_check_CD = jac_check[:, 2:]

    vsyst_check_AB = jac_check_AB @ jac_check_AB.T
    vsyst_check_CD = jac_check_CD @ jac_check_CD.T

    vsyst_check = jac_check0 @ vpar @ jac_check0.T
    vsyst_check_23 = vsyst_check
    if not correlated:
        vsyst_check = 2.0 * vsyst_check_AB + vsyst_check_CD  # because of using AB and A,B

    vfull_check_AB = vstat + vsyst_check_AB
    vfull_check_CD = vstat + vsyst_check_CD
    vfull_check = vstat + vsyst_check
    vfull_check_23 = vstat + vsyst_check_23

    if correlated:
        factors = {"d": 0.0, "f": 10000}
    else:
        factors = {"d": 0.0, "f": 100}
    rtol = finfo(dtype).resolution
    if not correlated:
        assert allclose(jac_A, jac_check_A, rtol=factors[dtype] * rtol)
        assert allclose(jac_B, jac_check_B, rtol=factors[dtype] * rtol)
    assert allclose(jac_AB, jac_check_AB, rtol=factors[dtype] * rtol)
    assert allclose(jac_CD, jac_check_CD, rtol=factors[dtype] * rtol)

    assert allclose(jac_ABCD_2, jac_check0, rtol=factors[dtype] * rtol)
    assert allclose(jac_ABCD_3, jac_check, rtol=factors[dtype] * rtol)

    assert allclose(vsyst_AB, vsyst_check_AB, rtol=factors[dtype] * rtol)
    if not correlated:
        assert allclose(vsyst_AcB, vsyst_check_AB, rtol=factors[dtype] * rtol)
    assert allclose(vsyst_CD, vsyst_check_CD, rtol=factors[dtype] * rtol)
    assert allclose(vsyst, vsyst_check, rtol=factors[dtype] * rtol)
    assert allclose(vsyst2, vsyst_check_23, rtol=factors[dtype] * rtol)
    assert allclose(vsyst3, vsyst_check_23, rtol=factors[dtype] * rtol)

    assert allclose(vfull_AB, vfull_check_AB, rtol=factors[dtype] * rtol)
    assert allclose(vfull_CD, vfull_check_CD, rtol=factors[dtype] * rtol)
    assert allclose(vfull, vfull_check, rtol=factors[dtype] * rtol)
    assert allclose(vfull2, vfull_check_23, rtol=factors[dtype] * rtol)
    assert allclose(vfull3, vfull_check_23, rtol=factors[dtype] * rtol)

    for jacs in cm._dict_jacobian.values():
        for jac in jacs:
            assert jac.frozen
            jac.outputs[0].set(-1)

    cm.update_matrices()

    jac_AB = cm._dict_jacobian["covmat AB"][0].get_data()
    jac_CD = cm._dict_jacobian["covmat CD"][0].get_data()

    assert allclose(jac_AB, jac_check_AB, rtol=factors[dtype] * rtol)
    assert allclose(jac_CD, jac_check_CD, rtol=factors[dtype] * rtol)

    graph.touch()
    savegraph(graph, f"output/{testname}.dot", show="full")
