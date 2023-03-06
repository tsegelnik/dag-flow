from typing import Tuple, TypeVar

GraphT = TypeVar("GraphT", bound="Graph")
NodeT = TypeVar("NodeT", bound="Node")
InputT = TypeVar("InputT", bound="Input")
InputsT = TypeVar("InputsT", bound="Inputs")
OutputT = TypeVar("OutputT", bound="Output")
OutputsT = TypeVar("OutputsT", bound="Outputs")

ShapeLike = Tuple[int, ...]
EdgesLike = Tuple[OutputT]
