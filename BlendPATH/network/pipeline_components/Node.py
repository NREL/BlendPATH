from dataclasses import dataclass, field

import cantera as ct

from . import Composition, Compressor, Pipe, eos
from . import cantera_util as ctu


@dataclass
class Node:
    """
    Nodes in a network. Must be connected to pipes or compressors. Can be a supply or demand as well.
    """

    name: str = ""
    p_max_mpa_g: float = 0
    index: int = 0
    pressure: float = None
    X: float = "CH4:1"
    connections: dict = field(default_factory=lambda: {"Pipe": [], "Comp": []})
    is_demand: bool = False

    def clear_connections(self) -> None:
        """
        Function to clear out connections
        """
        self.connections = {"Pipe": [], "Comp": []}

    def add_connection(self, cxn) -> None:
        """
        Add pipe or compressor cxn to node
        """
        if isinstance(cxn, Pipe.Pipe):
            self.connections["Pipe"].append(cxn)
        elif isinstance(cxn, Compressor.Compressor):
            self.connections["Comp"].append(cxn)
        else:
            raise ValueError(
                f"Connection provided {type(cxn)} was not Pipe or Compressor"
            )

    def update_state(self, T: float, p: float, X: Composition, eos_type: str) -> tuple:
        """
        Update the temperature, pressure, and composition at the node. Calculate rho and z based on EOS
        """
        p_abs = p + ct.one_atm
        ctu.gas.TPX = T, p_abs, X.x_str
        rho = ctu.gas.density  # kg/m3
        mw = ctu.gas.mean_molecular_weight  # g/mol
        self.mw = mw
        self.pressure = p
        self.cp = ctu.gas.cp_mass
        self.cv = ctu.gas.cv_mass

        rho, z = eos.get_rz(p_gauge=p, T_K=T, X=X, eos=eos_type, mw=mw)

        self.z = z
        self.rho = rho

        return rho, z

    def heating_value(self) -> float:
        """
        Get the higher heating value at the node
        """
        return self.X.HHV
