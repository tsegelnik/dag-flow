from typing import Tuple, TypeVar, Union

NodeT = TypeVar("NodeT", bound="Node")
InputT = TypeVar("InputT", bound="Input")
InputsT = TypeVar("InputsT", bound="Inputs")
OutputT = TypeVar("OutputT", bound="Output")
OutputsT = TypeVar("OutputsT", bound="Outputs")

ShapeLikeT = Union[Tuple[int, ...], None] # like DTypeLike from Numpy
EdgesLikeT = Union[Tuple[OutputT], None]
