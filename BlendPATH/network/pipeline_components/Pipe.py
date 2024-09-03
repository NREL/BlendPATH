from dataclasses import dataclass
from typing import TYPE_CHECKING

import cantera as ct
import numpy as np

import BlendPATH.Global as gl
import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.util.schedules import SCHEDULES

from . import cantera_util as ctu
from . import eos

if TYPE_CHECKING:
    from .Node import Node


@dataclass
class Pipe:
    """
    A pipe that connects two nodes
    """

    from_node: "Node"
    to_node: "Node"
    name: str = ""
    diameter_mm: float = 0
    length_km: float = 0
    roughness_mm: float = 0.012
    thickness_mm: float = 0
    grade: str = ""
    p_max_mpa_g: float = 0
    m_dot: float = None

    def __post_init__(self) -> None:
        self.diameter_out_mm = self.diameter_mm + 2 * self.thickness_mm
        self.A_m2 = (self.diameter_mm * gl.MM2M) ** 2 / 4 * np.pi
        self.assign_DN()
        self.assign_sch()

    def assign_DN(self) -> None:
        """
        Determine the DN of the pipe based on the SCHEDULE table
        """
        outer_diam_list = SCHEDULES["Outer diameter [mm]"].values
        dn_list = SCHEDULES["DN"].values
        self.DN = float(
            dn_list[np.digitize(self.diameter_out_mm, outer_diam_list, right=True)]
        )

    def assign_sch(self) -> None:
        """
        Return the schedule based on the DN and lookup in SCHEDULES table
        """
        offset = 2
        sch_row = SCHEDULES.loc[SCHEDULES["DN"] == self.DN].values[0]
        sch_row = sch_row[offset:]
        sch_ind = np.nanargmin(abs(sch_row - self.thickness_mm))
        self.schedule = SCHEDULES.columns.values[sch_ind + offset]

    def pipe_assessment(
        self,
        design_option: str = "b",
        location_class: int = 1,
        joint_factor: float = 1,
        T_derating_factor: float = 1,
    ) -> None:
        """
        Reassign the ASME B31.12 design pressure
        """
        self.SMYS, self.SMTS = bp_pa.get_SMYS_SMTS(self.grade)
        self.pressure_ASME_MPa = self.design_pressure_ASME(
            design_option=design_option,
            location_class=location_class,
            joint_factor=joint_factor,
            T_derating_factor=T_derating_factor,
        )

    def design_pressure_ASME(
        self,
        design_option: str,
        location_class: int,
        joint_factor: int,
        T_derating_factor: int,
    ) -> float:
        """
        Calculates the ASME B31.12 design pressure
        """
        design_factor = bp_pa.get_design_factor(
            design_option=design_option, location_class=location_class
        )
        pressure_ASME_MPa = bp_pa.get_design_pressure_ASME(
            design_p_MPa=self.p_max_mpa_g,
            design_option=design_option,
            SMYS=self.SMYS,
            SMTS=self.SMTS,
            t=self.thickness_mm,
            D=self.DN,
            F=design_factor,
            E=joint_factor,
            T=T_derating_factor,
        )
        return pressure_ASME_MPa

    def get_derivative(self, eos_type: eos._EOS_OPTIONS) -> tuple:
        """
        Calculate the derivative dm/dp for the solver
        """
        p_in = self.from_node.pressure
        p_out = self.to_node.pressure
        C_p_eqn = self.get_flow_eqn_const(derivative=True, eos_type=eos_type)

        return C_p_eqn, p_in, p_out

    def get_mdot(self, eos_type: eos._EOS_OPTIONS) -> float:
        """
        Get the mass flow rate through the pipe
        """
        C_p_eqn = self.get_flow_eqn_const(derivative=False, eos_type=eos_type)
        direction = self.get_direction()
        self.m_dot = C_p_eqn * direction
        return C_p_eqn * direction

    def get_flow_eqn_const(
        self, derivative: bool = True, eos_type: eos._EOS_OPTIONS = "rk"
    ) -> float:
        """
        Calculate momentum equation coefficient
        """
        p_in = self.from_node.pressure + ct.one_atm
        p_out = self.to_node.pressure + ct.one_atm

        A = self.A_m2
        mw = self.from_node.mw
        T = gl.T_FIXED
        D = self.diameter_mm * gl.MM2M
        L = self.length_km * gl.KM2M

        p_avg = 2 / 3 * (p_in + p_out - p_in * p_out / (p_in + p_out)) - ct.one_atm
        ctu.gas.TPX = T, p_avg + ct.one_atm, self.from_node.X.x_str
        mu = ctu.gas.viscosity

        rho_avg, z_avg = eos.get_rz(
            p_gauge=p_avg, T_K=T, X=self.from_node.X, eos=eos_type, mw=mw
        )

        if self.m_dot is None:
            Re = 1e8
        else:
            v_avg = abs(self.m_dot / rho_avg / self.A_m2)
            Re = rho_avg * v_avg * D / mu
            self.v_avg = v_avg
            self.v_from = abs(self.m_dot / self.from_node.rho / self.A_m2)
            self.v_to = abs(self.m_dot / self.to_node.rho / self.A_m2)
        self.Re = Re

        f = self.get_friction_factor(Re, self.roughness_mm, self.diameter_mm)
        self.f = f

        coef = A * (mw / z_avg / ctu.R_GAS / T * D / f / L) ** (0.5)

        p_eqn = abs(p_in**2 - p_out**2) ** (0.5 * (-1 if derivative else 1))

        return coef * p_eqn

    def get_friction_factor(self, Re: float, RO: float, D: float) -> float:
        """
        Use Hofer explicit approximation of Colebrook-White equation for friction factor
        """
        return (-2 * np.log10(4.518 / Re * np.log10(Re / 7) + RO / (3.71 * D))) ** (-2)

    def get_direction(self) -> float:
        """
        Return direction of pipe flow
        """
        if self.to_node.pressure > self.from_node.pressure:
            return -1
        return 1

    def get_mach_number(self) -> float:
        """
        Calculate mach number
        """
        v = [self.v_from, self.v_to]
        c = [
            np.sqrt(x.cp / x.cv * x.pressure / x.rho)
            for x in [self.from_node, self.to_node]
        ]
        m = [v[x] / c[x] for x in [0, 1]]
        return max(m)

    def get_erosional_velocity(self) -> float:
        """
        Calculate ASME B31.12 erosional velocity
        """
        KG2LB = 2.20462
        M32FT3 = 35.3147
        FT2M = 0.3048

        rho = min([x.rho for x in [self.from_node, self.to_node]])
        u = 100 / np.sqrt(rho * KG2LB / M32FT3)
        u_m_s = u * FT2M
        return u_m_s  # m/s
