from dataclasses import dataclass

import numpy as np

import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.util.schedules import SCHEDULES

from .Node import Node


@dataclass
class PipeSegment:
    """
    Pipe segment of a network. Collection of pipes. Also compressor is if it at the end of the segment
    """

    pipes: list
    diameter: float
    DN: float
    comps: list
    start_node: Node = None
    end_node: Node = None
    nodes: list = None
    demand_nodes: list = None
    length_km: float = 0
    pressure_violation: bool = False
    p_max_mpa_g: float = 0
    pressure_ASME_MPa: float = 0
    offtake_lengths: list = None

    def post_segmentation(self) -> None:
        """
        Assign overall parameters of the segment after segmentation process
        """
        self.assign_length()
        self.assign_end_node()
        self.assign_flows()

    def assign_length(self) -> None:
        """
        Calculate total length of segment
        """
        self.length_km = sum([pipe.length_km for pipe in self.pipes])

    def check_p_violations(self) -> None:
        """
        Determine if any pressure violations within pipe segment
        """
        self.pressure_violation = True in [pipe.design_violation for pipe in self.pipes]

    def assign_flows(self) -> None:
        """
        Inlet and outlet flow rate based on first and last pipe
        """
        self.mdot_in = self.pipes[0].m_dot
        self.mdot_out = self.pipes[-1].m_dot

    def assign_end_node(self) -> None:
        """
        Determine last node in the segment by taking the last node in the node list. Used when determined segment outlet pressure
        """
        self.end_node = self.nodes[-1]

    def get_DNs(self, max_number=5) -> tuple:
        """
        Get the DN options equal or larger than the existing pipe. Use in PL, DR method
        """
        pick_cols = SCHEDULES.loc[
            SCHEDULES["DN"] >= self.DN, ["DN", "Outer diameter [mm]"]
        ]
        pick_DNs = pick_cols["DN"].tolist()
        pick_ODs = pick_cols["Outer diameter [mm]"].tolist()
        dn_options = pick_DNs[:max_number]
        od_options = pick_ODs[:max_number]
        return dn_options, od_options

    def get_viable_schedules(
        self,
        design_option: bp_pa._DESIGN_OPTIONS,
        ASME_params: bp_pa.ASME_consts,
        grade: str,
        ASME_pressure_flag: bool = False,
        DN: float = None,
        return_all: bool = False,
    ) -> tuple:
        """
        Get viable schedules for the pipesegment
        """

        if DN is None:
            check_DN = self.DN
        else:
            check_DN = DN
        sch_list = (
            SCHEDULES.loc[SCHEDULES["DN"] == check_DN]
            .dropna(axis=1)
            .to_dict(orient="split")
        )

        design_pressure = (
            self.pressure_ASME_MPa if ASME_pressure_flag else self.p_max_mpa_g
        )

        (th, schedule, pressure, index) = bp_pa.get_viable_schedules(
            sch_list,
            design_option,
            ASME_params,
            grade,
            design_pressure,
            self.pressure_ASME_MPa,
            check_DN,
        )
        if return_all:
            return (th, schedule, pressure)
        else:
            if index == -1:
                return (None, np.nan, None)
            return (th[index], schedule[index], pressure[index])
