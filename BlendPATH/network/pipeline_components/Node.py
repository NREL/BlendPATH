from dataclasses import dataclass, field

import cantera as ct

import BlendPATH.Global as gl

from . import Composition, Compressor, Pipe
from . import cantera_util as ctu
from . import eos


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
    thermo_curvefit: bool = False

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

        self.mw = X.mw
        self.pressure = p

        if self.thermo_curvefit:
            rho, z = self.X.get_curvefit_rho_z(p_gauge_pa=p)
        else:
            rho, z = eos.get_rz(p_gauge=p, T_K=T, X=X, eos=eos_type, mw=X.mw)

        self.z = z
        self.rho = rho

        return rho, z

    def heating_value(self) -> float:
        """
        Get the higher heating value at the node
        """
        return self.X.HHV

    @property
    def cpcv(self):
        """
        Heat capacity
        """
        ctu.gas.TPX = gl.T_FIXED, self.pressure + ct.one_atm, self.X.x_str
        return ctu.gas.cp_mass / ctu.gas.cv_mass
