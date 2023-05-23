from typing import Literal, Mapping

from ..meta_node import MetaNode
from .Interpolator import Interpolator
from .SegmentIndex import SegmentIndex


def InterpolatorGroup(
    method: Literal["linear", "log", "logx", "exp"] = "linear",
    tolerance: float = 1e-10,
    underflow: Literal[
        "constant", "nearestedge", "extrapolate"
    ] = "extrapolate",
    overflow: Literal[
        "constant", "nearestedge", "extrapolate"
    ] = "extrapolate",
    fillvalue: float = 0.0,
    *,
    labels: Mapping = {}
) -> MetaNode:
    segmentIndex = SegmentIndex(
        "SegmentIndex", label=labels.get("segment index", {})
    )
    interpolator = Interpolator(
        "interpolator",
        method=method,
        tolerance=tolerance,
        underflow=underflow,
        overflow=overflow,
        fillvalue=fillvalue,
        label=labels.get("interpolator", {}),
    )
    segmentIndex.outputs["indices"] >> interpolator("indices")

    metaint = MetaNode()
    metaint.add_node(
        segmentIndex,
        kw_inputs=["coarse", "fine"],
        merge_inputs=["coarse", "fine"],
    )
    metaint.add_node(
        interpolator,
        kw_inputs=["coarse", "fine", "yc"],
        kw_outputs=["result"],
        merge_inputs=["coarse", "fine"],
        inputs_pos=True,
        outputs_pos=True,
        #missing_inputs=True,
        #also_missing_outputs=True,
    )

    return metaint
