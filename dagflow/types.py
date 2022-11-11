from typing import TypeVar
from weakref import ReferenceType

NodeT = TypeVar('NodeT', bound='Node')
InputT = TypeVar('InputT', bound='Input')
OutputT = TypeVar('OutputT', bound='Output')

NodeRefT = ReferenceType[NodeT]

