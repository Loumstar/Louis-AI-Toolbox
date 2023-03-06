from typing import Union

from .gru import GRUCell
from .lstm import LSTMCell
from .vanilla import VanillaCell

CellType = Union[GRUCell, LSTMCell, VanillaCell]
