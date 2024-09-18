from dataclasses import dataclass, field

import cantera as ct
import pandas as pd

import BlendPATH.Global as gl

from . import cantera_util as ctu

_t_k = 273.15
_CRIT_C = {
    "H2": (-239.95 + _t_k, 1297000),
    "CH4": (-82.595 + _t_k, 4598800),
    "C2H6": (32.68 + _t_k, 4880000),
    "C3H8": (96.67 + _t_k, 4250000),
    "CO2": (31.05 + _t_k, 7386000),
    "N2": (-146.95 + _t_k, 3390000),
    "C4H10": (151.99 + _t_k, 3784000),
    "C5H12": (196.54 + _t_k, 3364000),
    "iC4H10": (134.98 + _t_k, 3648000),
}


@dataclass
class Composition:
    """
    Gas-phase composition
    """

    pure_x: dict = field(default_factory=lambda: {})

    valid_vals = ["H2", "CH4", "C2H6", "C3H8", "CO2", "N2", "C4H10", "C5H12", "iC4H10"]

    def __post_init__(self):
        self.x = self.pure_x.copy()
        self.as_str()
        self.get_comp()
        self.calc_heating_value()

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """
        Create composition from a pandas df
        """
        species = df["SPECIES"].values.tolist()
        mole_frac = df["X"].values.tolist()

        composition = dict(zip(species, mole_frac))

        return Composition(composition)

    def blendH2(self, blend: float) -> None:
        """
        Update composition based on a new H2 mole fraction. Note that this requires no H2 in the original composition
        """
        if not (0 <= blend <= 1):
            raise ValueError(
                "Blend percent must be represented as a fraction between 0 and 1"
            )
        self.x = {a: b * (1 - blend) for a, b in self.pure_x.items() if a != "H2"}
        self.x["H2"] = blend
        self.as_str()
        self.get_comp()
        self.calc_heating_value()

    def just_fuel(self) -> str:
        """
        Get the fuel components of the gas, used in HHV calculation
        """
        not_fuels = ["CO2", "N2"]
        fuels_x = {x: v for x, v in self.x.items() if x not in not_fuels}
        new_total = sum(fuels_x.values())
        fuels_x = {x: v / new_total for x, v in fuels_x.items()}
        return str(fuels_x).strip("{}").replace(" ", "").replace("'", "")

    def as_str(self) -> None:
        """
        Assigns str value for input into Cantera
        """
        self.x_str = str(self.x).strip("{}").replace(" ", "").replace("'", "")

    def get_comp(self) -> None:
        """
        Assigns critical temperature and pressure
        """
        tc = 0
        pc = 0
        for i, v in self.x.items():
            tc += _CRIT_C[i][0] * v
            pc += _CRIT_C[i][1] * v

        self.tc = tc
        self.pc = pc

    def calc_heating_value(self) -> float:
        """
        Calculate higher heating value of mixture using Cantera in MJ/kg
        """
        ctu.gas.TPX = 298.15, ct.one_atm, self.x_str
        self.mw = ctu.gas.mean_molecular_weight
        mole_frac_fuel = 1 - ctu.gas["N2"].Y[0] - ctu.gas["CO2"].Y[0]
        jf_X = self.just_fuel()
        ctu.gas.set_equivalence_ratio(1.0, jf_X, "O2:1.0")
        h1 = ctu.gas.enthalpy_mass
        Y_fuel = 1 - ctu.gas["O2"].Y[0]

        X_products = {
            "CO2": ctu.gas.elemental_mole_fraction("C"),
            "H2O": 0.5 * ctu.gas.elemental_mole_fraction("H"),
            "N2": 0.5 * ctu.gas.elemental_mole_fraction("N"),
        }

        ctu.gas.TPX = None, None, X_products
        Y_H2O = ctu.gas["H2O"].Y[0]
        h2 = ctu.gas.enthalpy_mass
        # LHV = -(h2 - h1) / Y_fuel / 1e6
        HHV = -(h2 - h1 + ctu.h_water * Y_H2O) / Y_fuel / gl.MJ2J * mole_frac_fuel
        self.HHV = HHV
        return HHV  #  MJ/kg

    def get_GCV(self) -> float:
        """
        Return GCV in MJ/sm3
        """
        ctu.gas.TPX = 273.15, ct.one_atm, self.x_str
        mw = ctu.gas.mean_molecular_weight
        v = ctu.gas.volume_mole
        return self.HHV * mw / v
