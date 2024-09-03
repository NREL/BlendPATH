from dataclasses import dataclass

import BlendPATH.Global as gl

from .Demand_node import Demand_node


@dataclass
class Supply_node(Demand_node):
    """
    A supply node where pressure is fixed
    """

    is_pressure_supply: bool = True
    pressure_mpa: float = 0

    def __post_init__(self):
        self.node.pressure = self.pressure_mpa * gl.MPA2PA
