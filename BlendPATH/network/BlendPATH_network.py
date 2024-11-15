from dataclasses import dataclass
from os.path import isfile
from typing import get_args

import numpy as np
import pandas as pd

import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components.cantera_util as ctu
from BlendPATH.network.pipeline_components.eos import _EOS_OPTIONS
from BlendPATH.util.pipe_assessment import ASME_consts

from . import pipeline_components as plc


class BlendPATH_network:
    """
    Hydaulic network
    """

    def __init__(
        self,
        name: str = "",
        pipes: dict = None,
        nodes: dict = None,
        demand_nodes: dict = None,
        supply_nodes: dict = None,
        compressors: dict = None,
        composition: list = None,
        eos: _EOS_OPTIONS = "rk",
        thermo_curvefit: bool = False,
    ) -> None:
        """
        Initialize a network
        """
        # Assign defaults
        self.name = name
        self.pipes = pipes if pipes is not None else {}
        self.nodes = nodes if nodes is not None else {}
        self.demand_nodes = demand_nodes if demand_nodes is not None else {}
        self.supply_nodes = supply_nodes if supply_nodes is not None else {}
        self.compressors = compressors if compressors is not None else {}
        self.composition = composition
        self.set_thermo_curvefit(thermo_curvefit)
        self.set_eos(eos)

        self.assign_node_indices()
        self.assign_ignore_nodes()
        self.assign_connections()

        # Calculate network capacity
        self.capacity_MMBTU_day = sum(
            [d.flowrate_MMBTU_day for d in self.demand_nodes.values()]
        )
        # Check if need more segments if some pipes are too long
        # self.check_segmentation()

    def assign_connections(self) -> None:
        """
        Add connections for nodes to pipes/comps
        """
        for node in self.nodes.values():
            node.clear_connections()
        for pipe in self.pipes.values():
            pipe.from_node.add_connection(pipe)
            pipe.to_node.add_connection(pipe)
        for comp in self.compressors.values():
            comp.from_node.add_connection(comp)
            comp.to_node.add_connection(comp)

    def assign_node_indices(self) -> None:
        """
        Assigns an index to each node
        """
        node_index = 0
        for node in self.nodes.values():
            node.index = node_index
            node_index += 1

    def assign_ignore_nodes(self) -> None:
        """
        Adds supply nodes and compressor to_node to ignored nodes
        """
        self.ignore_nodes = []
        for sn in self.supply_nodes.values():
            self.ignore_nodes.append(sn.node.index)
        for comp in self.compressors.values():
            self.ignore_nodes.append(comp.to_node.index)
        self.ignore_nodes.sort()

    @classmethod
    def import_from_file(cls, name: str, subsegments: bool = False):
        """
        Make network from file
        """

        def check_node(node_list: dict, check_node: str) -> None:
            if check_node not in node_list:  # Check that the node exists
                raise ValueError(f"{check_node} is not in the node list")

        # Read in input from excel file by sheet
        filename = f"{name}"
        if not isfile(filename):
            raise ValueError(f"Could not find file with file name: {filename}")
        nodes_df = pd.read_excel(filename, "NODES")
        pipes_df = pd.read_excel(filename, "PIPES")
        compressors_df = pd.read_excel(filename, "COMPRESSORS")
        supply_df = pd.read_excel(filename, "SUPPLY")
        demand_df = pd.read_excel(filename, "DEMAND")
        composition_df = pd.read_excel(filename, "COMPOSITION")

        # Load in composition
        composition = plc.Composition.from_df(composition_df)

        # Create nodes
        nodes = {}
        for i, row in nodes_df.iterrows():
            nodes[row["node_name"]] = plc.Node(
                name=row["node_name"],
                p_max_mpa_g=row["p_max_mpa_g"],
                X=composition,
            )

        # Create supply nodes
        supply_nodes = {}
        for i, row in supply_df.iterrows():
            node = row["node_name"]
            check_node(nodes, node)
            is_pressure_supply = not pd.isna(row["pressure_mpa_g"])
            # Default to pressure boundary if supplied
            supply_nodes[row["supply_name"]] = plc.Supply_node(
                name=row["supply_name"],
                node=nodes[row["node_name"]],
                is_pressure_supply=is_pressure_supply,
                pressure_mpa=row["pressure_mpa_g"],
                flowrate_MW=row["flowrate_MW"],
            )

            supply_nodes[row["supply_name"]].node.pressure = (
                row["pressure_mpa_g"] * gl.MPA2PA
            )

        # Create demand nodes
        demand_nodes = {}
        for i, row in demand_df.iterrows():
            node = row["node_name"]
            check_node(nodes, node)
            demand_nodes[row["demand_name"]] = plc.Demand_node(
                name=row["demand_name"],
                node=nodes[node],
                flowrate_MW=row["flowrate_MW"],
            )

        # Create pipes
        pipes = {}
        for i, row in pipes_df.iterrows():
            from_node = row["from_node"]
            to_node = row["to_node"]
            check_node(nodes, from_node)
            check_node(nodes, to_node)
            max_pressure_mpa = min(
                [nodes[from_node].p_max_mpa_g, nodes[from_node].p_max_mpa_g]
            )

            pipes[row["pipe_name"]] = plc.Pipe(
                name=row["pipe_name"],
                from_node=nodes[from_node],
                to_node=nodes[to_node],
                diameter_mm=row["diameter_mm"],
                length_km=row["length_km"],
                roughness_mm=row["roughness_mm"],
                thickness_mm=row["thickness_mm"],
                grade=row["steel_grade"],
                p_max_mpa_g=max_pressure_mpa,
            )

        # Compressors
        compressors = {}
        for i, row in compressors_df.iterrows():
            from_node = row["from_node"]
            to_node = row["to_node"]
            check_node(nodes, from_node)
            check_node(nodes, to_node)
            compressors[row["compressor_name"]] = plc.Compressor(
                name=row["compressor_name"],
                from_node=nodes[from_node],
                to_node=nodes[to_node],
                pressure_out_mpa_g=row["pressure_out_mpa_g"],
                original_rating_MW=row["rating_MW"],
                fuel_extract=row["extract_fuel"],
            )
            # Assign efficiency to gas or electric based on if fuel extraction is true/false
            if not np.isnan(row["eta_s"]):
                if not 0 < row["eta_s"] <= 1:
                    raise ValueError(
                        f"Compressor efficiency eta_s must be a fraction, {row['eta_s']} was given"
                    )
                if row["extract_fuel"]:
                    compressors[row["compressor_name"]].eta_comp_s = float(row["eta_s"])
                else:
                    compressors[row["compressor_name"]].eta_comp_s_elec = float(
                        row["eta_s"]
                    )
            if not np.isnan(row["eta_driver"]):
                if not 0 < row["eta_driver"] <= 1:
                    raise ValueError(
                        f"Compressor efficiency eta_driver must be a fraction, {row['eta_driver']} was given"
                    )
                if row["extract_fuel"]:
                    compressors[row["compressor_name"]].eta_driver = float(
                        row["eta_driver"]
                    )
                else:
                    compressors[row["compressor_name"]].eta_driver_elec = float(
                        row["eta_driver"]
                    )
            pass

        return BlendPATH_network(
            name=name,
            pipes=pipes,
            nodes=nodes,
            demand_nodes=demand_nodes,
            supply_nodes=supply_nodes,
            compressors=compressors,
            composition=composition,
        )

    def initialize(self, supply_node: plc.Supply_node, cr_max: float) -> np.ndarray:
        """
        Set initial guess of pressures
        """
        p_init = np.zeros(self.n_nodes)
        # Get the first supply node - as a starting point
        p_supply_node = list(self.supply_nodes.values())[0]
        p_init[p_supply_node.node.index] = p_supply_node.pressure_mpa * gl.MPA2PA
        p_supply_node.node.update_state(
            T=gl.T_FIXED,
            p=p_supply_node.pressure_mpa * gl.MPA2PA,
            X=self.composition,
            eos_type=self.eos,
        )
        for comp in self.compressors.values():
            comp.to_node.update_state(
                T=gl.T_FIXED,
                p=comp.pressure_out_mpa_g * gl.MPA2PA,
                X=self.composition,
                eos_type=self.eos,
            )

        # Reset all nodes besides supply node, compressor to nodes to unknown pressure
        comp_to_nodes = [x.to_node for x in self.compressors.values()]
        for n in self.nodes.values():
            if (n is p_supply_node.node) or n in comp_to_nodes:
                continue
            n.pressure = None

        def get_p_out(
            p_in: float,
            m_dot: float,
            A: float,
            mw: float,
            zrt: float,
            d: float,
            f: float,
            L: float,
        ) -> float:
            """
            Calculate the outlet pressure (for initialization)
            """
            c = (m_dot / (A * (mw / zrt * d / f / L) ** 0.5)) ** 2
            if c > p_in**2:
                return p_in * 0.95
            return (p_in**2 - c) ** 0.5

        def get_new_node(n: plc.Node, p: plc.Pipe) -> plc.Node:
            """
            Find the next node connected to the pipe that isnt the current node
            """
            if p.to_node is n:
                return p.from_node
            elif p.from_node is n:
                return p.to_node
            else:
                raise ValueError("Can't find node")

        def init_pressure(
            up_node: plc.Node, new_node: plc.Node, pipe: plc.Pipe
        ) -> None:
            """
            Initialized pressures at nodes. Runs recursively
            """
            if new_node.pressure is not None:
                return

            m_dot_in = m_dot_demands / len(up_node.connections["Pipe"]) - (
                1 if up_node is p_supply_node.node else 0
            )
            p_out = get_p_out(
                p_in=up_node.pressure,
                m_dot=m_dot_in,
                A=np.pi * (pipe.diameter_mm * gl.MM2M) ** 2 / 4,
                mw=up_node.mw,
                zrt=1 * ctu.gas_constant * gl.T_FIXED,
                d=pipe.diameter_mm * gl.MM2M,
                f=0.01,
                L=pipe.length_km * gl.KM2M,
            )
            if p_out == up_node.pressure:
                p_out *= 0.99
            new_node.update_state(
                T=gl.T_FIXED,
                p=p_out,
                X=self.composition,
                eos_type=self.eos,
            )
            for pipe_next in new_node.connections["Pipe"]:
                if pipe_next is pipe:
                    continue
                new_node_next = get_new_node(new_node, pipe_next)
                if new_node_next.pressure is not None:
                    continue
                init_pressure(new_node, new_node_next, pipe_next)

        m_dot_demands = sum(
            [demand.flowrate_mdot for demand in self.demand_nodes.values()]
        )
        for pipe in p_supply_node.node.connections["Pipe"]:
            init_pressure(
                supply_node.node, get_new_node(p_supply_node.node, pipe), pipe
            )
        for comp in self.compressors.values():
            for pipe in comp.to_node.connections["Pipe"]:
                init_pressure(comp.to_node, get_new_node(comp.to_node, pipe), pipe)

        # Preset compressor inlet to be based on CR
        for comp in self.compressors.values():
            if comp.name == "Supply compressor":
                continue
            comp.from_node.pressure = comp.to_node.pressure / cr_max

        for n in self.nodes.values():
            p_init[n.index] = n.pressure
        return p_init

    def initialize_depreciated(self, supply_node: plc.Supply_node) -> np.ndarray:
        """
        DEPRECIATED:
        Initialize network pressure for solving and returns array of pressures
        """
        # Initialization
        p_init = np.zeros(self.n_nodes)
        # Get the first supply node - as a starting point
        p_supply_node = list(self.supply_nodes.values())[0]
        p_init[p_supply_node.node.index] = p_supply_node.pressure_mpa * gl.MPA2PA

        # Reset all nodes besides supply node, compressor to nodes to unknown pressure
        for comp in self.compressors.values():
            comp.to_node.pressure = comp.pressure_out_mpa_g * gl.MPA2PA
        comp_to_nodes = [x.to_node for x in self.compressors.values()]
        for n in self.nodes.values():
            if (n is p_supply_node.node) or n in comp_to_nodes:
                continue
            n.pressure = None

        # Recursively initialize pressure
        def init_pressure(upstream_n: plc.Node, node: plc.Node, is_comp=False) -> None:
            # Multiply by random factor
            factor = 0.99 * (1 - np.random.rand() / 100)

            # If pressure is already set, break out unless it is purposedly set
            if node.pressure is not None and not is_comp:
                p_init[node.index] = node.pressure
                return

            # Make downstream node 99% of upstrea
            if not is_comp:
                node.pressure = upstream_n.pressure * factor
            p_init[node.index] = node.pressure

            # Loop through downstream connection nodes
            for pipe in node.connections["Pipe"]:
                next_node = pipe.to_node
                if next_node is upstream_n:
                    next_node = pipe.from_node
                if next_node is not node:
                    init_pressure(node, next_node)
            for comp in node.connections["Comp"]:
                next_node = comp.to_node
                if next_node is upstream_n:
                    next_node = comp.from_node
                if next_node is not node:
                    init_pressure(
                        node,
                        next_node,
                        True,
                    )

        for pipe in p_supply_node.node.connections["Pipe"]:
            # Get the next node. It will be either the to or from node of the pipe connection
            next_node = pipe.to_node
            if next_node is supply_node:
                next_node = pipe.from_node

            init_pressure(supply_node.node, next_node)

        for comp in p_supply_node.node.connections["Comp"]:
            init_pressure(comp.from_node, comp.to_node)
            # Get the next node. It will be either the to or from node of the pipe connection
            for pipe in comp.to_node.connections["Pipe"]:
                next_node = pipe.to_node
                if next_node is supply_node:
                    next_node = pipe.from_node

                init_pressure(supply_node.node, next_node)

        # for comp in self.compressors.values():
        #     from_node = comp.from_node
        #     p_init[from_node.index] = comp.to_node.pressure/1.5

        p_index = 0
        for node in self.nodes.values():
            node.update_state(
                T=gl.T_FIXED, p=p_init[p_index], X=self.composition, eos_type=self.eos
            )
            p_index = p_index + 1

        return p_init

    def blendH2(self, blend: float) -> None:
        """
        Blend amount of H2. Reassigns composition and recalculated flow rates
        """
        self.composition.blendH2(blend)
        # Update demand node conversion from MW to kg/s
        for dn in self.demand_nodes.values():
            dn.recalc_mdot()
        self.reassign_offtakes()

    def solve(self, c_relax: float = gl.RELAX_FACTOR, cr_max: float = 1.5) -> None:
        """
        Solve network pressures
        """
        n_nodes = len(self.nodes)
        self.n_nodes = n_nodes
        supply_node = list(self.supply_nodes.values())[0]

        m_dot_target = np.zeros(n_nodes)
        for node in self.demand_nodes.values():
            m_dot_target[node.node.index] += node.flowrate_mdot
        m_dot_sum = np.sum(m_dot_target)
        # This sums up all other flows( assuming only demand)
        m_dot_target[supply_node.node.index] = -1 * m_dot_sum

        # Initialize pressure and set the state of each node
        p_init = self.initialize(supply_node, cr_max=cr_max)

        # Loop
        p_solving = p_init
        n_iter = 0
        err = np.inf
        while err > gl.SOLVER_TOL:
            jacobian, m_dot = self.make_jacobian()
            nodal_flow = np.dot(m_dot, np.ones(n_nodes))

            delta_flow = m_dot_target - nodal_flow
            delta_flow = np.delete(delta_flow, self.ignore_nodes)

            delta_p = np.linalg.solve(jacobian, delta_flow)

            # Resizes so known values are skipped
            for i in self.ignore_nodes:
                delta_p = np.insert(delta_p, i, 0)

            p_solving += delta_p / c_relax

            if np.any(p_solving < 0):
                ind = np.where(p_solving < 0)
                p_solving[ind] -= (delta_p[ind] / c_relax) * 0.99
                if np.any(p_solving < 0):
                    raise ValueError("Negative pressure")

            # Assumes everything is ordered
            p_index = 0
            for node in self.nodes.values():
                node.update_state(
                    gl.T_FIXED,
                    p_solving[p_index],
                    self.composition,
                    eos_type=self.eos,
                )
                p_index = p_index + 1

            err = np.max(np.absolute(delta_flow))

            n_iter += 1

            for comp in self.compressors.values():
                to_c = comp.to_node.index
                from_c = comp.from_node.index
                comp_flow = -1 * nodal_flow[to_c]
                if to_c == n_nodes - 1:
                    # If there is a compressor at the end of the segment
                    comp_flow = m_dot_sum
                m_dot_target[from_c] = comp_flow
                # If using fuel extraction, then reduce mdot after comp
                fuel_use = comp.get_fuel_use(comp_flow)
                if not comp.fuel_extract:
                    continue
                m_dot_target[from_c] += fuel_use
                m_dot_target[supply_node.node.index] -= fuel_use

            if n_iter > gl.MAX_ITER:
                raise ValueError(f"Could not converge in {gl.MAX_ITER} iterations")

        # Chcck for any values below minimum pressure
        if np.any(gl.MIN_PRES - p_solving >= 20000):
            raise ValueError("Pressure below threshold")

        # Assign maximum pressure to pipe based on calculated node pressure
        for pipe in self.pipes.values():
            from_node_p = p_solving[pipe.from_node.index]
            to_node_p = p_solving[pipe.to_node.index]
            max_node_p = max(from_node_p, to_node_p)
            pipe.pressure_MPa = max_node_p / gl.MPA2PA

        # Assign demand flow rates
        # Some demands stack at a single node, so use same ratio of setpoints
        # to distribute calculated demand flow rate
        stacking_demand = {}
        for dnode in self.demand_nodes.values():
            if dnode.node.index not in stacking_demand.keys():
                stacking_demand[dnode.node.index] = {dnode.name: dnode.flowrate_mdot}
            else:
                stacking_demand[dnode.node.index][dnode.name] = dnode.flowrate_mdot
        for dnode in self.demand_nodes.values():
            dnode.flowrate_mdot_sim = (
                nodal_flow[dnode.node.index]
                * stacking_demand[dnode.node.index][dnode.name]
                / sum(stacking_demand[dnode.node.index].values())
            )

    def make_jacobian(self) -> tuple:
        """
        Make jacobian matrix for solver
        """
        len_all_nodes = len(self.nodes)
        len_p_supply_nodes = 0
        len_comps = 0

        # Only need to solve to unknown pressures
        # P supplies are known and pressure outlet of compressors is known
        n_nodes = len_all_nodes - len_p_supply_nodes - len_comps

        # Initialize jacobian and mass flow rate arrays
        jacobian = np.zeros((n_nodes, n_nodes))
        m_dot = np.zeros((n_nodes, n_nodes))

        # Loop thru pipes
        for pipe in self.pipes.values():
            to_node = pipe.to_node
            from_node = pipe.from_node
            to_node_index = to_node.index
            from_node_index = from_node.index

            # Assign based on connections and ignore nodes
            dm_dp, p_in, p_out = pipe.get_derivative(self.eos)
            if (
                to_node_index not in self.ignore_nodes
                and from_node_index not in self.ignore_nodes
            ):
                jacobian[to_node_index][from_node_index] += dm_dp * p_in
                jacobian[from_node_index][to_node_index] += dm_dp * p_out
            if to_node_index not in self.ignore_nodes:
                jacobian[to_node_index][to_node_index] -= dm_dp * p_out
            if from_node_index not in self.ignore_nodes:
                jacobian[from_node_index][from_node_index] -= dm_dp * p_in

            # Assign mass flow rates
            m_dot_pipe = pipe.get_mdot(self.eos)
            m_dot[to_node_index, from_node_index] += m_dot_pipe
            m_dot[from_node_index, to_node_index] -= m_dot_pipe

        # Remove nodes that don't need tracking
        jacobian = np.delete(jacobian, self.ignore_nodes, 0)
        jacobian = np.delete(jacobian, self.ignore_nodes, 1)

        return jacobian, m_dot

    def segment_pipe(self) -> list:
        """
        Segment pipeline network into segments based on compressors, branches, diameter change
        """
        p_supply_node = list(self.supply_nodes.values())[0]
        pipe_segments = []
        segment_i = 0

        # Recursive function to get segments
        def get_segments(node, segment_i: int) -> None:
            # Flag for it this segment is new
            new_flag = False
            # Loop thru pipe connections of this node
            for pipe in node.connections["Pipe"]:
                # If segments were increased then make a new segment
                if len(pipe_segments) <= segment_i:
                    pipe_segments.append(
                        plc.PipeSegment(
                            pipes=[pipe],
                            diameter=pipe.diameter_mm,
                            DN=pipe.DN,
                            comps=[],
                            start_node=node,
                            p_max_mpa_g=pipe.p_max_mpa_g,
                            pressure_ASME_MPa=pipe.pressure_ASME_MPa,
                            nodes=[node],
                            demand_nodes=[],
                            offtake_lengths=[pipe.length_km],
                        )
                    )
                    new_flag = True
                # If the pipe is already in the added pipes, then only add if the node is not added.
                # This is only the case when coming to the end of a pipeline
                elif pipe in pipe_segments[segment_i].pipes:
                    if node not in pipe_segments[segment_i].nodes:
                        pipe_segments[segment_i].nodes.append(node)
                    if (
                        node.is_demand
                        and node not in pipe_segments[segment_i].demand_nodes
                    ):
                        pipe_segments[segment_i].demand_nodes.append(node)
                        pipe_segments[segment_i].offtake_lengths.append(0)

                    continue

                # check diameter
                if pipe.diameter_mm == pipe_segments[segment_i].diameter:
                    if not new_flag:
                        pipe_segments[segment_i].pipes.append(pipe)
                        if node not in pipe_segments[segment_i].nodes:
                            pipe_segments[segment_i].nodes.append(node)
                        pipe_segments[segment_i].offtake_lengths[-1] += pipe.length_km
                        if node.is_demand:
                            if node not in pipe_segments[segment_i].demand_nodes:
                                pipe_segments[segment_i].demand_nodes.append(node)
                                pipe_segments[segment_i].offtake_lengths.append(0)

                    new_node = pipe.to_node
                    if new_node == node:
                        new_node = pipe.from_node
                    get_segments(new_node, segment_i)
                else:
                    segment_i += 1
                    pipe_segments.append[
                        plc.PipeSegment(
                            pipes=[pipe],
                            diameter=pipe.diameter_mm,
                            DN=pipe.DN,
                            comps=[],
                            start_node=node,
                            p_max_mpa_g=pipe.p_max_mpa_g,
                            pressure_ASME_MPa=pipe.pressure_ASME_MPa,
                            nodes=[node],
                            demand_nodes=[],
                            offtake_lengths=[pipe.length_km],
                        )
                    ]
                    new_node = pipe.to_node
                    if new_node == node:
                        new_node = pipe.from_node
                    get_segments(new_node, segment_i)

            for comp in node.connections["Comp"]:
                if comp in pipe_segments[segment_i - 1].comps:
                    continue
                new_node = comp.to_node
                if new_node == node:
                    new_node = comp.from_node
                for pipe in new_node.connections["Pipe"]:
                    pipe_segments[segment_i].comps.append(comp)
                    segment_i += 1
                    get_segments(new_node, segment_i)

        # Start at supply node
        get_segments(p_supply_node.node, segment_i)

        return pipe_segments

    def pipe_assessment(
        self, ASME_params: ASME_consts, design_option: str = "b"
    ) -> None:
        """
        Assess pipe MAOP based on ASME B31.12
        """
        for pipe in self.pipes.values():
            pipe.pipe_assessment(
                design_option=design_option,
                location_class=ASME_params.location_class,
                joint_factor=ASME_params.joint_factor,
                T_derating_factor=ASME_params.T_rating,
            )

    def reassign_offtakes(self) -> None:
        """
        Assign mass flow rates and HHV based on changes in composition
        """
        if not hasattr(self, "pipe_segments"):
            return
        for ps in self.pipe_segments:
            ps.offtake_mdots = []
            ps.HHV = ps.start_node.heating_value()

            for demand_node in ps.demand_nodes:
                ps.offtake_mdots.append(
                    np.sum(
                        [
                            d_n.flowrate_mdot
                            for d_n in self.demand_nodes.values()
                            if d_n.node is demand_node
                        ]
                    )
                )
            ps.post_segmentation()
        return

    def set_eos(self, eos: _EOS_OPTIONS = "rk") -> None:
        """
        Set equation of state. Loops through nodes and pipes
        """
        eos_lc = eos.lower()
        if eos_lc not in get_args(_EOS_OPTIONS):
            raise ValueError(
                f"{eos} is not a valid design option (must be one of {list(get_args(_EOS_OPTIONS))})"
            )
        self.eos = eos
        for node in self.nodes.values():
            node.eos = eos
        for pipe in self.pipes.values():
            pipe.eos = eos

    def check_segmentation(self) -> None:
        """
        Unused function to check if a pipe needs further segmentation based on L/D ratio
        """
        seg_max = 30000
        node_index = len(self.nodes.keys()) - 1
        all_pipe_names = list(self.pipes.keys())
        for pipe_name in all_pipe_names:
            pipe = self.pipes[pipe_name]
            if (pipe.length_km * gl.KM2M) / (pipe.diameter_mm * gl.MM2M) > seg_max:
                lenth_sub_segment = seg_max * (pipe.diameter_mm * gl.MM2M) / gl.KM2M
                n_nodes = int(np.floor(pipe.length_km / lenth_sub_segment))
                from_node = pipe.from_node
                from_node.connections["Pipe"].remove(pipe.name)
                for subseg in range(n_nodes):
                    new_node_name = f"{from_node.name}_{subseg}"
                    # Keep addng underscript till it is a unique name
                    while new_node_name in self.nodes.keys():
                        new_node_name = f"{new_node_name}_"
                    self.nodes[new_node_name] = plc.Node(
                        name=new_node_name,
                        p_max_mpa_g=from_node.p_max_mpa_g,
                        index=node_index + 1,
                        X=from_node.X,
                    )
                    node_index += 1
                    new_pipe_name = f"{pipe.name}_{subseg}"
                    while new_pipe_name in self.pipes.keys():
                        new_pipe_name = f"{new_pipe_name}_"
                    to_node = self.nodes[new_node_name]
                    self.pipes[new_pipe_name] = plc.Pipe(
                        name=new_pipe_name,
                        from_node=from_node,
                        to_node=to_node,
                        diameter_mm=pipe.diameter_mm,
                        length_km=lenth_sub_segment,
                        roughness_mm=pipe.roughness_mm,
                        thickness_mm=pipe.thickness_mm,
                        grade=pipe.grade,
                        p_max_mpa_g=pipe.p_max_mpa_g,
                    )
                    from_node = to_node

                length_remaining = pipe.length_km % lenth_sub_segment
                pipe.from_node = from_node
                pipe.length_km = length_remaining

    def to_file(self, filename: str) -> None:
        """
        Export hydraulic model results to file
        """
        # Specify output writer
        writer = pd.ExcelWriter(filename, engine="xlsxwriter")
        workbook = writer.book

        # Info
        sheetname = "Info"
        worksheet = workbook.add_worksheet(sheetname)
        info_out = []
        info_out.append(("Network name", self.name))
        info_out.append(("EOS", self.eos))

        info_out = pd.DataFrame(info_out, columns=["Name", "Pressure (Pa)"])
        startrow = 0
        info_out.to_excel(
            writer, sheet_name=sheetname, startrow=startrow, startcol=0, index=False
        )

        # Nodes
        sheetname = "Nodes"
        worksheet = workbook.add_worksheet(sheetname)
        writer.sheets[sheetname] = worksheet
        nodes_out = []
        for node in self.nodes.values():
            nodes_out.append((node.name, node.pressure))
        nodes_out = pd.DataFrame(nodes_out, columns=["Name", "Pressure (Pa)"])
        startrow = 0
        nodes_out.to_excel(
            writer, sheet_name=sheetname, startrow=startrow, startcol=0, index=False
        )

        # Pipes
        sheetname = "Pipes"
        worksheet = workbook.add_worksheet(sheetname)
        writer.sheets[sheetname] = worksheet
        pipes_out = []
        for pipe in self.pipes.values():
            pipes_out.append(
                (
                    pipe.name,
                    pipe.from_node.name,
                    pipe.to_node.name,
                    pipe.m_dot,
                    pipe.length_km,
                    pipe.from_node.pressure,
                    pipe.to_node.pressure,
                    max([pipe.v_from, pipe.v_to]),
                    pipe.Re,
                    pipe.f,
                )
            )
        pipes_out = pd.DataFrame(
            pipes_out,
            columns=[
                "Name",
                "From node",
                "To node",
                "Flow rate (kg/s)",
                "Length (km)",
                "From node pressure (Pa)",
                "To node pressure (Pa)",
                "Max velocity (m/s)",
                "Reynolds number",
                "Friction factor",
            ],
        )
        startrow = 0
        pipes_out.to_excel(
            writer, sheet_name=sheetname, startrow=startrow, startcol=0, index=False
        )

        # Comps
        sheetname = "Compressors"
        worksheet = workbook.add_worksheet(sheetname)
        writer.sheets[sheetname] = worksheet
        comps_out = []
        for comp in self.compressors.values():
            comps_out.append(
                (
                    comp.name,
                    comp.from_node.name,
                    comp.to_node.name,
                    comp.get_cr_ratio(),
                    comp.get_fuel_use_MMBTU_hr(),
                    comp.shaft_power_MW,
                    comp.shaft_power_MW * gl.MW2HP,
                    comp.eta_comp_s,
                    comp.eta_driver,
                )
            )
        comps_out = pd.DataFrame(
            comps_out,
            columns=[
                "Name",
                "From node",
                "To node",
                "Pressure Ratio",
                "Fuel Consumption [MMBTU/hr]",
                "Shaft power [MW]",
                "Shaft power [hp]",
                "Isentropic efficiency",
                "Mechanical efficiency",
            ],
        )
        startrow = 0
        comps_out.to_excel(
            writer, sheet_name=sheetname, startrow=startrow, startcol=0, index=False
        )

        # Composition
        sheetname = "Composition"
        worksheet = workbook.add_worksheet(sheetname)
        writer.sheets[sheetname] = worksheet
        composition_out = []
        for i, v in self.composition.x.items():
            composition_out.append((i, v))
        composition_out = pd.DataFrame(
            composition_out, columns=["Name", "Molar fraction"]
        )
        startrow = 0
        composition_out.to_excel(
            writer, sheet_name=sheetname, startrow=startrow, startcol=0, index=False
        )

        ### Closeout
        writer._save()

    def set_thermo_curvefit(self, thermo_curvefit: bool) -> None:
        """
        Set thermo curvefit for nodes
        """
        self.thermo_curvefit = thermo_curvefit
        for node in self.nodes.values():
            node.thermo_curvefit = thermo_curvefit
        for pipe in self.pipes.values():
            pipe.thermo_curvefit = thermo_curvefit


@dataclass
class Design_params:
    final_outlet_pressure_mpa_g: float
    max_CR: list
    existing_comp_elec: bool
    new_comp_elec: bool
    new_comp_eta_s: float
    new_comp_eta_s_elec: float
    new_comp_eta_driver: float
    new_comp_eta_driver_elec: float
