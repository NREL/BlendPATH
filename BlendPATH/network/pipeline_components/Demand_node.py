from dataclasses import dataclass

import BlendPATH.Global as gl

from .Node import Node


@dataclass
class Demand_node:
    """
    A network node that has a specified energy flow rate
    """

    node: Node
    name: str = ""
    flowrate_MW: float = 0

    def __post_init__(self) -> None:
        self.recalc_mdot()
        self.flowrate_MMBTU_day = self.flowrate_MW * gl.MW2MMBTUDAY
        self.node.is_demand = True

    def recalc_mdot(self) -> None:
        """
        Calculate the new flow rate based on the HHV
        """
        hhv = self.node.heating_value()
        self.flowrate_mdot = self.flowrate_MW / hhv  # kg/s
