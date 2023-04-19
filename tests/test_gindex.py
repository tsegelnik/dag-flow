from itertools import product

import pytest
from gindex.gindex import GIndex, GIndexInstance, GIndexName
from gindex.gnindex import GNIndex, GNIndexInstance

@pytest.fixture
def detector_name() -> GIndexName:
    return GIndexName("det", "detector")


@pytest.fixture
def subdetector_name() -> GIndexName:
    return GIndexName("subdet", "subdetector")


@pytest.fixture
def index_values() -> tuple:
    return ("01", "02", "03")


@pytest.fixture
def detector(detector_name, index_values) -> GIndex:
    return GIndex(detector_name, index_values)


@pytest.fixture
def subdetector(subdetector_name, index_values) -> GIndex:
    return GIndex(subdetector_name, index_values)


@pytest.fixture
def detector01(detector) -> GIndexInstance:
    return detector[0]


def test_gindex(detector, subdetector, detector01):
    assert detector[0] == detector01
    assert detector01.copywith(value="02") == detector[1]
    assert id(detector01.name) == id(detector[0].name)
    assert all(detector.name == x.name for x in detector)
    assert all(subdetector.name == x.name for x in subdetector)
    assert detector01.copy() == detector01
    assert (
        detector.copywith(name=subdetector.name, values=subdetector.values)
        == subdetector
    )
    assert (
        subdetector.copywith(name=detector.name, values=detector.values)
        == detector
    )


def test_gindex_format(detector01, detector_name):
    assert all(
        x == "" for x in (detector01.formatwith(""), detector01.format(""))
    )
    assert all(
        x == "_01"
        for x in (
            detector01.formattedwith(),
            detector01.formatted(),
            detector01.formatwith("{here}", place="here"),
        )
    )
    assert (
        detector01.formatwith(
            "back{nind}",
            sep="|",
            withname=True,
            namemode="f",
            namesep=".",
            place="nind",
        )
        == "back|detector.01"
    )
    for name in detector_name:
        assert detector01.format("{" f"{name}" "}") == "_01"
        assert detector01.format("back{" f"{name}" "}") == "back_01"
        assert (
            detector01.formatwith("back{" f"{name}" "}", sep=".") == "back.01"
        )
        assert (
            detector01.formatwith("back{" f"{name}" "}", withname=True)
            == "back_det_01"
        )
        assert (
            detector01.formatwith(
                "back{" f"{name}" "}", withname=True, namesep="."
            )
            == "back_det.01"
        )
        assert (
            detector01.formatwith(
                "back{" f"{name}" "}", withname=True, namemode="f"
            )
            == "back_detector_01"
        )
        assert (
            detector01.formatwith(
                "back{" f"{name}" "}",
                sep="|",
                withname=True,
                namemode="f",
                namesep=".",
            )
            == "back|detector.01"
        )


def test_gnindex(detector, subdetector):
    nind = GNIndexInstance(values=(detector[0], subdetector[0]))
    assert nind
    assert nind.format(string="Spectrum") == (
        f"Spectrum{detector[0].sep}{detector[0].value}"
        f"{subdetector[0].sep}{subdetector[0].value}"
    )
    assert nind.formatwith(string="Spectrum", order=("det", -1, "subdet")) == (
        f"{detector[0].sep}{detector[0].value}Spectrum"
        f"{subdetector[0].sep}{subdetector[0].value}"
    )
    assert nind.formatwith(string="Spectrum", order=("subdet", "det", -1)) == (
        f"{subdetector[0].sep}{subdetector[0].value}"
        f"{detector[0].sep}{detector[0].value}Spectrum"
    )
    assert nind.formatwith(string="Spectrum", order=("det", -1)) == (
        f"{detector[0].sep}{detector[0].value}Spectrum"
    )


def test_gnindex_iter(detector, subdetector, index_values):
    sep = "_"
    nind = GNIndex(values=(detector, subdetector), sep=sep)
    nvals = tuple(
        sep.join(pair) for pair in product(index_values, index_values)
    )
    for i, inst in enumerate(nind):
        assert isinstance(inst, GNIndexInstance)
        assert inst.formattedwith(sep=sep) == f"{sep}{nvals[i]}"


def test_gnindex_arithmetic(detector, subdetector):
    gorder = ("det", "subdet", "i")
    nind = GNIndex(values=(detector, subdetector), order=gorder)
    ind = GIndex(GIndexName("i", "index"), ("1", "2"))
    nind2 = GNIndex(values=(detector, ind), order=gorder)
    nind3 = GNIndex(values=(ind,), order=gorder)
    # `sub` and `-`
    assert all(x - x == x.copywith(values=tuple()) for x in (nind, nind2))
    assert all(
        x.sub(("new",)) == x.copywith(values=tuple()) for x in (nind, nind2)
    )
    assert all(x.sub(x.names1d()) == x for x in (nind, nind2))
    assert nind2.sub(("i",)) == nind.copywith(values=(ind,))
    # `merge` and  `+`
    assert all(
        len(x.values) == len(nind.values)
        and set(x.values) == set(nind.values)
        and x.order == gorder
        for x in (nind + nind, nind | nind, nind.union(nind))
    )
    assert all(
        (y := nind + nind2) and y == x and y.order == gorder
        for x in (
            nind.copywith(values={detector, subdetector, ind}),
            nind2.copywith(values={detector, subdetector, ind}),
            nind.union(nind3),
            nind | nind2,
        )
    )


def test_gnindex_rest_split(
    detector, subdetector, detector_name, subdetector_name
):
    gorder = ("det", "subdet", "i")
    iname = GIndexName("i", "index")
    ind = GIndex(iname, ("1", "2"))
    nind = GNIndex(values=(detector, subdetector, ind), order=gorder)
    # test `dict`
    assert all(
        x in nind.dict
        for x in (
            iname,
            detector_name,
            subdetector_name,
            "i",
            "index",
            *detector_name,
            *subdetector_name,
        )
    )
    # test `rest`
    for elem in (
        nind.rest(val)
        for val in ("det", "detector", ("det",), ("detector",), detector_name)
    ):
        assert isinstance(elem, GNIndex)
        assert elem.order == nind.order
        assert elem.values == (subdetector, ind)
    for elem in (
        nind.rest(val) for val in (iname, "i", "index", ("i",), ("index",))
    ):
        assert isinstance(elem, GNIndex)
        assert elem.order == nind.order
        assert elem.values == (detector, subdetector)
    # test `split`
    assert nind, None == nind.split(nind.names1d())
    assert nind.copywith(values=tuple()), nind == nind.split(tuple())
    for elem, rest in (
        nind.split(val) for val in (("det",), ("detector",), (detector_name,))
    ):
        assert isinstance(elem, GNIndex) and isinstance(rest, GNIndex)
        assert elem.order == nind.order and rest.order == nind.order
        assert elem.values == (detector,) and rest.values == (subdetector, ind)
    for elem, rest in (
        nind.split(val)
        for val in (
            ("subdet",),
            ("subdetector",),
            (subdetector_name,),
        )
    ):
        assert isinstance(elem, GNIndex) and isinstance(rest, GNIndex)
        assert elem.order == nind.order and rest.order == nind.order
        assert elem.values == (subdetector,) and rest.values == (detector, ind)
    for elem, rest in (
        nind.split(val)
        for val in (
            ("detector", "subdet"),
            ("det", "subdetector"),
            (detector_name, subdetector_name),
        )
    ):
        assert isinstance(elem, GNIndex) and isinstance(rest, GNIndex)
        assert elem.order == nind.order and rest.order == nind.order
        assert elem.values == (detector, subdetector) and rest.values == (ind,)


def test_gnindex_order_exception(detector, subdetector, detector_name):
    orders = (object, 12, {4, 3, 2}, detector_name, detector)
    with pytest.raises(TypeError):
        for order in orders:
            GNIndexInstance(values=(detector[0], subdetector[0]), order=order)  # type: ignore
            GNIndex(values=(detector, subdetector), order=order)  # type: ignore
