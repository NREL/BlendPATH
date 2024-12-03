from dataclasses import dataclass, field

import cantera as ct
import numpy as np
import pandas as pd
import scipy.interpolate

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
    interp: bool = True

    valid_vals = ["H2", "CH4", "C2H6", "C3H8", "CO2", "N2", "C4H10", "C5H12", "iC4H10"]

    def __post_init__(self):
        self.x = self.pure_x.copy()
        self.as_str()
        self.get_comp()
        self.calc_heating_value()
        if self.interp:
            self.make_linear_interp()

    @classmethod
    def from_df(cls, df: pd.DataFrame, interp: bool = True):
        """
        Create composition from a pandas df
        """
        species = df["SPECIES"].values.tolist()
        mole_frac = df["X"].values.tolist()

        composition = dict(zip(species, mole_frac))

        return Composition(composition, interp)

    def blendH2(self, blend: float) -> None:
        """
        Update composition based on a new H2 mole fraction. Note that this requires no H2 in the original composition
        """
        # Return early if blend is already set
        if "H2" in self.x.keys() and self.x["H2"] == blend:
            return
        if not (0 <= blend <= 1):
            raise ValueError(
                "Blend percent must be represented as a fraction between 0 and 1"
            )
        self.x = {a: b * (1 - blend) for a, b in self.pure_x.items() if a != "H2"}
        self.x["H2"] = blend
        self.as_str()
        self.get_comp()
        self.calc_heating_value()
        if self.interp:
            self.make_linear_interp()

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

    def make_linear_interp(self):
        p_val_len = 50
        p_vals = np.linspace(1, 20 * gl.MPA2PA, p_val_len)
        rho_vals = []
        mu_vals = []
        h_vals = []
        s_vals = []

        p_vals_final = np.linspace(1 * gl.MPA2PA, 20 * gl.MPA2PA, p_val_len)
        p_vals_final_2d = np.linspace(1 * gl.MPA2PA, 20 * gl.MPA2PA, p_val_len)

        ctu.gas.TPX = gl.T_FIXED, p_vals[0] + ct.one_atm, self.x_str
        s_low = ctu.gas.s
        ctu.gas.TPX = gl.T_FIXED, p_vals[-1] + ct.one_atm, self.x_str
        s_high = ctu.gas.s
        s_range_len = 50
        s_range = np.linspace(s_high, s_low, s_range_len)
        h_2d_vals = np.zeros((p_val_len, s_range_len))

        for p_i, p in enumerate(p_vals):
            p_a = p + ct.one_atm
            ctu.gas.TPX = gl.T_FIXED, p_a, self.x_str
            while ~np.isfinite(ctu.gas.density):
                p_a *= 1.0001
                ctu.gas.TPX = gl.T_FIXED, p_a, self.x_str
            p_vals_final[p_i] = p_a

            rho_vals.append(ctu.gas.density)
            mu_vals.append(ctu.gas.viscosity)
            h_vals.append(ctu.gas.h)
            s_vals.append(ctu.gas.s)

            for s_i, s in enumerate(s_range):
                p_a = p + ct.one_atm
                while h_2d_vals[p_i, s_i] == 0:
                    try:
                        ctu.gas.SPX = s, p_a, self.x_str
                    except ct.CanteraError:
                        p_a *= 1.0001
                        ctu.gas.SPX = s, p_a, self.x_str
                    else:
                        h_2d_vals[p_i, s_i] = ctu.gas.h
                p_vals_final_2d[p_i] = p_a

        self.curve_fit_rho = (p_vals_final / gl.MPA2PA, np.array(rho_vals))
        self.curve_fit_mu = (p_vals_final / gl.MPA2PA, np.array(mu_vals))
        self.curve_fit_h = (p_vals_final / gl.MPA2PA, np.array(h_vals))
        self.curve_fit_s = (p_vals_final / gl.MPA2PA, np.array(s_vals))
        self.curve_fit_h_2d = scipy.interpolate.RectBivariateSpline(
            p_vals_final_2d / gl.MPA2PA, s_range, h_2d_vals, kx=1, ky=1
        )

    def get_curvefit_rho_z(self, p_gauge_pa: np.ndarray) -> np.ndarray:
        p_abs_pa = p_gauge_pa + ct.one_atm
        p_gauge_mpa = p_gauge_pa / gl.MPA2PA

        rho = np.interp(p_gauge_mpa, self.curve_fit_rho[0], self.curve_fit_rho[1])
        z = p_abs_pa * self.mw / ctu.R_GAS / gl.T_FIXED / rho
        if np.any(rho < 0):
            raise ValueError("Negative pressure")
        return rho, z

    def get_curvefit_mu(self, p_gauge_pa: np.ndarray) -> np.ndarray:
        p_gauge_mpa = p_gauge_pa / gl.MPA2PA

        mu = np.interp(p_gauge_mpa, self.curve_fit_mu[0], self.curve_fit_mu[1])
        return mu

    def get_curvefit_h(self, p_gauge_pa: np.ndarray) -> np.ndarray:
        p_gauge_mpa = p_gauge_pa / gl.MPA2PA

        return np.interp(p_gauge_mpa, self.curve_fit_h[0], self.curve_fit_h[1])

    def get_curvefit_s(self, p_gauge_pa: np.ndarray) -> np.ndarray:
        p_gauge_mpa = p_gauge_pa / gl.MPA2PA

        return np.interp(p_gauge_mpa, self.curve_fit_s[0], self.curve_fit_s[1])

    def get_curvefit_h_2d(self, p_gauge_pa: np.ndarray, s: np.ndarray) -> np.ndarray:
        p_gauge_mpa = p_gauge_pa / gl.MPA2PA
        return self.curve_fit_h_2d.ev(p_gauge_mpa, s)
