from typing import Literal, get_args

import cantera as ct
import numpy as np

from . import Composition
from . import cantera_util as ctu

_EOS_OPTIONS = Literal["rk", "papay"]


def get_rz(
    p_gauge: float = 101325,
    T_K: float = 273.15,
    X: Composition = None,
    eos: _EOS_OPTIONS = "rk",
    mw: float = 0,
) -> tuple:
    """
    Function to return the density and compressibility depending on EOS
    """
    eos_lc = eos.lower()
    if eos_lc not in get_args(_EOS_OPTIONS):
        raise ValueError(
            f"{eos} is not a valid design option (must be one of {list(get_args(_EOS_OPTIONS))})"
        )
    if eos_lc == "papay":
        return eos_papay(p_gauge, T_K, X.pc, X.tc, mw)
    if eos_lc == "rk":
        return eos_rk(p_gauge, T_K, X.x_str, mw)


def eos_papay(p_gauge: float, T_K: float, pc: float, tc: float, mw: float) -> tuple:
    """
    Use papay equation of state, not recommended for high hydrogen concentrations
    """
    p_abs = p_gauge + ct.one_atm

    P_r = p_abs / pc
    T_r = T_K / tc
    z = (
        1
        - (3.53 * P_r / (10 ** (0.9813 * T_r)))
        + (0.274 * P_r**2 / (10 ** (0.8157 * T_r)))
    )
    rho = p_abs / ctu.R_GAS / T_K / z * mw

    return rho, z


def eos_rk(p_gauge: float, T_K: float, X: str, mw: float) -> tuple:
    """
    Use Redlich-Kwong equation of state - calculated in Cantera
    """
    p_abs = p_gauge + ct.one_atm

    ctu.gas.TPX = T_K, p_abs, X
    rho = ctu.gas.density

    if np.isnan(rho) or rho == np.inf:
        ctu.gas.TPX = T_K, p_abs * 1.05, X
        rho_1 = ctu.gas.density
        ctu.gas.TPX = T_K, p_abs * 0.95, X
        rho_2 = ctu.gas.density
        rho = (rho_1 + rho_2) / 2
    z = p_abs * mw / ctu.R_GAS / T_K / rho

    return rho, z
