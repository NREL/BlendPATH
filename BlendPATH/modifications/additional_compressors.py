import copy
import math

import cantera as ct
import numpy as np
from pandas import DataFrame, ExcelWriter

import BlendPATH.costing.costing as bp_cost
import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components.cantera_util as ctu
from BlendPATH.network import pipeline_components as bp_plc
from BlendPATH.network.BlendPATH_network import BlendPATH_network, Design_params


def additional_compressors(
    network: BlendPATH_network,
    design_params: Design_params = None,
    new_filename: str = "modified",
    costing_params: bp_cost.Costing_params = None,
) -> tuple:
    """
    Modify network with additional compressors method
    """

    # Copy the network
    nw = copy.deepcopy(network)
    # Convert compressors. If not converting to electric, see keep the original choice
    for cs in nw.compressors.values():
        cs.fuel_extract = cs.fuel_extract and not design_params.existing_comp_elec

    # Set compression ratio variables
    max_CR = design_params.max_CR
    n_cr = len(max_CR)
    # Set final outlet pressure
    final_outlet_pressure = design_params.final_outlet_pressure_mpa_g

    # For new compressors (adl compressors + supply compressor)
    assign_eta_s = (
        design_params.new_comp_eta_s_elec
        if design_params.new_comp_elec
        else design_params.new_comp_eta_s
    )
    assign_eta_driver = (
        design_params.new_comp_eta_driver_elec
        if design_params.new_comp_elec
        else design_params.new_comp_eta_driver
    )

    # Copy gas composition
    composition = nw.composition
    # Sum up all demands
    demands_MW = [demand.flowrate_MW for demand in nw.demand_nodes.values()]

    # Save number of pipe segments
    n_ps = len(nw.pipe_segments)

    # Initialize lists

    # cs_fuel_use = [0] * n_ps
    # cs_fuel_use_elec = [0] * n_ps

    cr_lcot = [[] for _ in range(n_cr)]

    n_comps_ps_cr = [[] for _ in range(n_cr)]
    l_comps_ps_cr = [[] for _ in range(n_cr)]
    add_supply_comp_list = [False] * n_cr
    add_supply_comp_list_final = [False] * n_cr
    inlet_p_result = [[] for _ in range(n_cr)]

    # Loop thru compression ratios
    for cr_i, CR_ratio in enumerate(max_CR):

        n_comps_ps_cr[cr_i] = [0] * n_ps
        l_comps_ps_cr[cr_i] = [[] for _ in range(n_ps)]
        cr_lcot[cr_i] = [0] * n_ps

        # Loop thru segments (in reverse)
        prev_ASME_pressure = -1
        m_dot_in_prev = nw.pipe_segments[-1].mdot_out
        for ps_i, ps in reversed(list(enumerate(nw.pipe_segments))):

            # Setup the list to loop thru supply pressures
            supp_p_list = [ps.pressure_ASME_MPa]
            # Only relevant for first segment
            if ps_i == 0:
                og_pressure = nw.supply_nodes[
                    list(nw.supply_nodes.keys())[0]
                ].pressure_mpa

                if og_pressure < ps.pressure_ASME_MPa:
                    supp_p_list = (
                        [og_pressure]
                        + list(
                            range(
                                math.ceil(og_pressure),
                                math.floor(ps.pressure_ASME_MPa),
                                1,
                            )
                        )
                        + [ps.pressure_ASME_MPa]
                    )

            pressure_in_MPa = ps.pressure_ASME_MPa
            pressure_in_Pa = pressure_in_MPa * gl.MPA2PA
            pressure_out_MPa = (prev_ASME_pressure / gl.MPA2PA) / CR_ratio
            # If segment is the last segment, the outlet pressure is specified
            if ps_i == n_ps - 1:
                pressure_out_MPa = final_outlet_pressure
            pressure_out_Pa = pressure_out_MPa * gl.MPA2PA
            ps_nodes = [x.name for x in ps.nodes]

            # Calculate all offtakes -- adds the pipe segment outlet as a
            # offtake if it is not already
            all_mdot = ps.offtake_mdots.copy()
            if (
                len(ps.offtake_mdots) == 0
                or abs(ps.offtake_mdots[-1] - m_dot_in_prev) / ps.offtake_mdots[-1]
                > 0.01
            ):
                all_mdot.append(m_dot_in_prev)
            else:
                all_mdot[-1] = m_dot_in_prev

            # Loop through supply pressures
            supp_p_min_res = []
            for sup_p in supp_p_list:
                comp_cost = []
                revamped_comp_capex = []
                supply_comp_capex = 0
                supply_comp_fuel = {"gas": 0, "elec": 0}

                # Get number of compressors and the lengths
                (
                    n_comps,
                    l_comps,
                    addl_cs_fuel_ps,
                    addl_cs_elec_ps,
                    m_dot_seg,
                    addl_comps,
                ) = get_num_compressors(
                    composition=composition,
                    p_in=sup_p * gl.MPA2PA,
                    p_out=pressure_out_Pa,
                    offtakes=ps.offtake_lengths,
                    offtakes_mdot=all_mdot,
                    d=ps.diameter,
                    l_total=ps.length_km,
                    cr_max=CR_ratio,
                    roughness_mm=ps.pipes[0].roughness_mm,
                    eta_s=assign_eta_s,
                    eta_driver=assign_eta_driver,
                    new_comps_elec=design_params.new_comp_elec,
                    eos=nw.eos,
                    thermo_curvefit=nw.thermo_curvefit,
                    comp_p_out=pressure_in_Pa,
                    seg_compressor=ps.comps,
                    prev_ASME_pressure=prev_ASME_pressure,
                    comps_elec=design_params.existing_comp_elec,
                )
                if n_comps == np.inf:
                    segment_lcot = np.inf
                else:

                    # Unit change for segment capacity and compressor fuel
                    capacity = sum(all_mdot) * ps.HHV * gl.MW2MMBTUDAY
                    new_pipe_cap = 0  # No pipes built
                    fuel_use = sum(addl_cs_fuel_ps) * ps.HHV * gl.MW2MMBTUDAY
                    elec_use = sum(addl_cs_elec_ps) * gl.DAY2HR / gl.KW2W

                    # Get compressor costs
                    for cs in addl_comps.values():
                        comp_cost.append(cs.get_cap_cost(cp=costing_params))
                        revamped_comp_capex.append(
                            cs.get_cap_cost(
                                cp=costing_params,
                                revamp=True,
                                to_electric=design_params.existing_comp_elec,
                            )
                        )

                    # Check if supply compressor needed
                    sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]
                    orig_supply_pressure = min(sn.pressure_mpa, ps.pressure_ASME_MPa)
                    if sn.node.name in ps_nodes and orig_supply_pressure < sup_p:
                        add_supply_comp_list[cr_i] = True
                        from_node = bp_plc.Node(
                            name="comp_to_node",
                            X=sn.node.X,
                            pressure=orig_supply_pressure * gl.MPA2PA,
                        )
                        supply_comp = bp_plc.Compressor(
                            name="new_supply_comp",
                            from_node=from_node,
                            to_node=bp_plc.Node(name="comp_to_node", X=sn.node.X),
                            pressure_out_mpa_g=sup_p,
                            fuel_extract=not design_params.new_comp_elec,
                        )

                        comp_h_1 = sn.node.X.get_curvefit_h(
                            supply_comp.from_node.pressure
                        )
                        comp_s_1 = sn.node.X.get_curvefit_s(
                            supply_comp.from_node.pressure
                        )
                        comp_h_2_s = sn.node.X.get_curvefit_h_2d(
                            supply_comp.to_node.pressure, comp_s_1
                        )
                        supply_comp_fuel = {
                            "gas": supply_comp.get_fuel_use(
                                comp_h_1, comp_s_1, comp_h_2_s, m_dot=m_dot_seg
                            )
                            * ps.HHV
                            * gl.MW2MMBTUDAY
                            / capacity,
                            "elec": supply_comp.fuel_electric_W
                            * gl.DAY2HR
                            / gl.KW2W
                            / capacity,
                        }

                        supply_comp_capex += supply_comp.get_cap_cost(cp=costing_params)
                    else:
                        add_supply_comp_list[cr_i] = False

                    meter_cost = bp_cost.meter_reg_station_cost(
                        cp=costing_params, demands_MW=demands_MW
                    )
                    # Zero since no pipe is added
                    ili_costs = 0
                    valve_cost = 0

                    if sum(comp_cost) + supply_comp_capex + fuel_use + elec_use == 0:
                        segment_lcot = 0
                    else:
                        price_breakdown = bp_cost.calc_lcot(
                            json_file=costing_params.casestudy_name,
                            capacity=capacity,
                            new_pipe_cap=new_pipe_cap,
                            comp_cost=comp_cost,
                            revamped_comp_capex=revamped_comp_capex,
                            supply_comp_capex=supply_comp_capex,
                            compressor_fuel=fuel_use / capacity,
                            compressor_fuel_elec=elec_use / capacity,
                            supply_comp_fuel=supply_comp_fuel,
                            cs_cost=costing_params.cf_price,
                            elec_cost=costing_params.elec_price,
                            meter_cost=meter_cost,
                            ili_costs=ili_costs,
                            valve_cost=valve_cost,
                            original_network_residual_value=costing_params.original_pipeline_cost,
                            financial_overrides=costing_params.financial_overrides,
                        )
                        segment_lcot = price_breakdown[
                            "LCOT: Levelized cost of transport"
                        ]

                    supp_p_min_res.append(
                        (
                            segment_lcot,
                            n_comps,
                            l_comps,
                            sup_p,
                            add_supply_comp_list[cr_i],
                        )
                    )

            # Get minimum solution per supply pressure
            lcot_p_spply = [x[0] for x in supp_p_min_res]
            idxmin_lcot_p_supp = lcot_p_spply.index(min(lcot_p_spply))
            # Asssign values for lowest LCOT per supply pressure
            cr_lcot[cr_i][ps_i] = supp_p_min_res[idxmin_lcot_p_supp][0]
            n_comps_ps_cr[cr_i][ps_i] = supp_p_min_res[idxmin_lcot_p_supp][1]
            l_comps_ps_cr[cr_i][ps_i] = supp_p_min_res[idxmin_lcot_p_supp][2]
            if ps_i == 0:
                add_supply_comp_list_final[cr_i] = supp_p_min_res[idxmin_lcot_p_supp][4]
                inlet_p_result[cr_i] = supp_p_min_res[idxmin_lcot_p_supp][3]

            # Assign flow rate and pressure to next segment
            m_dot_in_prev = m_dot_seg
            prev_ASME_pressure = pressure_in_Pa

    # Get the results for the CR with the lowest LCOT across segments
    cr_lcot_sums = [sum(x) for x in cr_lcot]
    cr_min_index = cr_lcot_sums.index(min(cr_lcot_sums))
    n_comps_ps = n_comps_ps_cr[cr_min_index]
    l_comps_ps = l_comps_ps_cr[cr_min_index]
    add_supply_comp = add_supply_comp_list_final[cr_min_index]
    supply_p = inlet_p_result[cr_min_index]

    # REMAKE file
    col_names = [
        "pipe_name",
        "from_node",
        "to_node",
        "diameter_mm",
        "length_km",
        "roughness_mm",
        "thickness_mm",
        "steel_grade",
    ]
    new_pipes = {x: [] for x in col_names}

    col_names = [
        "compressor_name",
        "from_node",
        "to_node",
        "pressure_out_mpa_g",
        "rating_MW",
        "extract_fuel",
        "eta_s",
        "eta_driver",
    ]
    new_comps = {x: [] for x in col_names}

    # Add nodes
    col_names = ["node_name", "p_max_mpa_g"]
    new_nodes = {x: [] for x in col_names}

    for ps_i, ps_comp in enumerate(n_comps_ps):
        # Get segment
        ps = nw.pipe_segments[ps_i]
        p_max_seg = ps.pressure_ASME_MPa

        # Get lengths of segment
        l_comps = l_comps_ps[ps_i]

        # Nodes
        len_added = 0
        comp_len_i = 0
        pipe_len_cum = 0
        for pipe in ps.pipes:
            pipe_len_cum += pipe.length_km
            pipe_len_remaining = pipe_len_cum

            pipe_segmented = False

            pipe_from_node = pipe.from_node.name
            new_nodes["node_name"].append(pipe_from_node)
            new_nodes["p_max_mpa_g"].append(p_max_seg)

            # If length passes where the compresser, add the compressor
            while comp_len_i < ps_comp and l_comps[comp_len_i] < pipe_len_remaining:
                pipe_segmented = True

                # Check if it overlaps with already existing node:
                if abs(l_comps[comp_len_i] - len_added) < 0.01:
                    from_comp_name = new_nodes["node_name"][-1]

                else:
                    # Add node before compressor
                    from_comp_name = f"N_pre_C_{ps_i}_{comp_len_i}"
                    new_nodes["node_name"].append(from_comp_name)
                    new_nodes["p_max_mpa_g"].append(p_max_seg)

                    # Add pipe to compressor
                    pipe_name = f"{pipe.name}_pre_C_{ps_i}_{comp_len_i}"
                    new_pipes["pipe_name"].append(pipe_name)
                    new_pipes["from_node"].append(pipe_from_node)
                    new_pipes["to_node"].append(from_comp_name)  #
                    new_pipes["length_km"].append(l_comps[comp_len_i] - len_added)
                    new_pipes["roughness_mm"].append(pipe.roughness_mm)
                    new_pipes["diameter_mm"].append(pipe.diameter_mm)
                    new_pipes["thickness_mm"].append(pipe.thickness_mm)
                    new_pipes["steel_grade"].append(pipe.grade)

                # Add node after compressor
                to_comp_name = f"N_post_C_{ps_i}_{comp_len_i}"
                new_nodes["node_name"].append(to_comp_name)
                new_nodes["p_max_mpa_g"].append(p_max_seg)

                # Add compressor
                comp_name = f"C_{ps_i}_{comp_len_i}"
                new_comps["compressor_name"].append(comp_name)
                new_comps["from_node"].append(from_comp_name)
                new_comps["to_node"].append(to_comp_name)
                new_comps["pressure_out_mpa_g"].append(p_max_seg)
                new_comps["rating_MW"].append(0)
                new_comps["extract_fuel"].append(not design_params.new_comp_elec)
                new_comps["eta_s"].append(assign_eta_s)
                new_comps["eta_driver"].append(assign_eta_driver)

                # Update latest node to the outlet of new compressor
                pipe_from_node = to_comp_name

                #
                len_added = l_comps[comp_len_i]
                # increase comp number
                comp_len_i += 1

            # Else add the pipe as usual
            else:
                pipe_name = pipe.name
                from_node = pipe.from_node.name
                pipe_len_final = pipe.length_km
                if pipe_segmented:
                    pipe_name = f"{pipe.name}_remaining"
                    from_node = pipe_from_node
                    pipe_len_final = pipe_len_remaining - len_added
                if pipe_len_final > 0.01:
                    # Add pipes as normal
                    new_pipes["pipe_name"].append(pipe_name)
                    new_pipes["from_node"].append(from_node)
                    new_pipes["to_node"].append(pipe.to_node.name)
                    new_pipes["length_km"].append(pipe_len_final)
                    new_pipes["roughness_mm"].append(pipe.roughness_mm)
                    new_pipes["diameter_mm"].append(pipe.diameter_mm)
                    new_pipes["thickness_mm"].append(pipe.thickness_mm)
                    new_pipes["steel_grade"].append(pipe.grade)
                    # Update total pipe length
                    len_added += pipe_len_final
                else:
                    new_nodes["node_name"].pop()
                    new_nodes["p_max_mpa_g"].pop()
                    # new_pipes["to_node"][-1] = pipe.to_node.name
                    new_comps["to_node"][-1] = pipe.to_node.name
                    pass

        new_nodes["node_name"].append(pipe.to_node.name)
        new_nodes["p_max_mpa_g"].append(p_max_seg)

    # Add existing compressors
    for comp in nw.compressors.values():
        new_comps["compressor_name"].append(comp.name)
        new_comps["from_node"].append(comp.from_node.name)
        new_comps["to_node"].append(comp.to_node.name)
        new_comps["rating_MW"].append(comp.original_rating_MW)
        new_comps["extract_fuel"].append(comp.fuel_extract)
        new_comps["eta_s"].append(
            comp.eta_comp_s if comp.fuel_extract else comp.eta_comp_s_elec
        )
        new_comps["eta_driver"].append(
            comp.eta_driver
            if comp.fuel_extract
            else np.nan  # comp.eta_driver_elec_used
        )

        # Assume to_node has to be the outlet pressure
        p_max = np.inf
        for pipe in comp.to_node.connections["Pipe"]:
            for key, ps in enumerate(nw.pipe_segments):
                if pipe in ps.pipes:
                    p_max = min(p_max, ps.pressure_ASME_MPa)

        new_comps["pressure_out_mpa_g"].append(p_max)

    # Add supply
    col_names = ["supply_name", "node_name", "pressure_mpa_g", "flowrate_MW"]
    new_supply = {x: [] for x in col_names}
    for supply in nw.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)

        # Assume to_node has to be the outlet pressure
        # p_max = np.inf

        # for pipe in supply.node.connections["Pipe"]:
        #     p_max = min(p_max, pipe.pressure_ASME_MPa)

        new_supply["pressure_mpa_g"].append(supply_p)
        new_supply["flowrate_MW"].append("")

    if add_supply_comp:
        sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]

        # Add new compressor
        new_comps["compressor_name"].insert(0, "Supply compressor")
        new_comps["from_node"].insert(0, "Supply compressor from_node")
        new_comps["to_node"].insert(0, sn.node.name)
        new_comps["rating_MW"].insert(0, 0)
        new_comps["extract_fuel"].insert(0, not design_params.new_comp_elec)
        new_comps["eta_s"].insert(0, assign_eta_s)
        new_comps["eta_driver"].insert(0, assign_eta_driver)
        p_max = np.inf
        for pipe in supply.node.connections["Pipe"]:
            p_max = min(pipe.pressure_ASME_MPa, p_max)
        new_comps["pressure_out_mpa_g"].insert(0, supply_p)

        # Add new node
        new_nodes["node_name"].insert(0, "Supply compressor from_node")
        new_nodes["p_max_mpa_g"].insert(0, p_max)

        # Update supply node
        new_supply["node_name"][-1] = "Supply compressor from_node"
        new_supply["pressure_mpa_g"][-1] = sn.pressure_mpa

    # Make new demands
    col_names = ["demand_name", "node_name", "flowrate_MW"]
    new_demand = {x: [] for x in col_names}
    for demand in nw.demand_nodes.values():
        new_demand["demand_name"].append(demand.name)
        new_demand["node_name"].append(demand.node.name)
        new_demand["flowrate_MW"].append(demand.flowrate_MW)

    # Composition
    col_names = ["SPECIES", "X"]
    new_composition = {x: [] for x in col_names}
    for species, x in nw.composition.x.items():
        new_composition["SPECIES"].append(species)
        new_composition["X"].append(x)

    new_pipes = DataFrame(new_pipes)
    new_nodes = DataFrame(new_nodes)
    new_comps = DataFrame(new_comps)
    new_supply = DataFrame(new_supply)
    new_demand = DataFrame(new_demand)
    new_composition = DataFrame(new_composition)

    # Sort compressors by node name order
    new_comps.from_node = new_comps.from_node.astype("category")
    new_comps.from_node = new_comps.from_node.cat.set_categories(
        new_nodes.node_name.values
    )
    new_comps = new_comps.sort_values(by="from_node")

    # Remake file
    with ExcelWriter(new_filename) as writer:
        new_pipes.to_excel(writer, sheet_name="PIPES", index=False)
        new_nodes.to_excel(writer, sheet_name="NODES", index=False)
        new_comps.to_excel(writer, sheet_name="COMPRESSORS", index=False)
        new_supply.to_excel(writer, sheet_name="SUPPLY", index=False)
        new_demand.to_excel(writer, sheet_name="DEMAND", index=False)
        new_composition.to_excel(writer, sheet_name="COMPOSITION", index=False)

    return n_comps_ps, l_comps_ps


def make_compressor_network(
    n_comps: int,
    composition: bp_plc.Composition,
    p_in: float,
    offtakes: list,
    all_mdot: list,
    d_main: float,
    l_total: float,
    roughness_mm: float,
    eta_s: float,
    eta_driver: float,
    comp_p_out: float,
    seg_compressor: list,
    prev_ASME_pressure: float,
    comps_elec: bool,
    eos: bp_plc.eos._EOS_OPTIONS = "rk",
    new_comps_elec: bool = True,
    thermo_curvefit: bool = False,
) -> tuple:
    """
    Make a new network with compressors added to segment
    """
    # Make inlet node
    n_ds_in = bp_plc.Node(name="in", X=composition, index=0)
    nodes = {n_ds_in.name: n_ds_in}
    prev_node = n_ds_in
    prev_length = 0

    pipes = {}
    compressors = {}
    demands = {}
    supplys = {
        "supply": bp_plc.Supply_node(node=n_ds_in, pressure_mpa=p_in / gl.MPA2PA)
    }

    HHV = composition.HHV

    l_between = l_total / (n_comps + 1)
    l_comps = [l_between * (x + 1) for x in range(n_comps)]

    # Make combined dict of all relevant compressor and offtake lengths
    comps_lengths = {x: {"val_type": "comp"} for x in l_comps}
    offtake_lengths = {
        x: {"val_type": "offtake", "mdot": all_mdot[i]}
        for i, x in enumerate(np.cumsum(offtakes))
    }
    # THIS ASSUMES UNIQUE VALUES - OFFTAKE AND COMPRESSOR CANNOT COEXIST CURRENTLY
    # all_lengths = comps_lengths | offtake_lengths
    all_lengths = {**comps_lengths, **offtake_lengths}
    # Then sort
    all_lengths = dict(sorted(all_lengths.items()))

    node_index = 1  # Since supply node was alread added
    pipe_index = 0
    demand_index = 0
    comp_index = 0
    for length, val in all_lengths.items():
        if val["val_type"] == "offtake":
            name = f"ot_{node_index-1}"
            nodes[name] = bp_plc.Node(name=name, X=composition)
            node_index += 1
            d_name = f"demand_{demand_index}"
            demands[d_name] = bp_plc.Demand_node(
                node=nodes[name],
                flowrate_MW=val["mdot"] * HHV,
            )
            demand_index += 1
            # MAKE PIPE
            p_name = f"pipe_{pipe_index}"
            pipes[p_name] = bp_plc.Pipe(
                name=p_name,
                from_node=prev_node,
                to_node=nodes[name],
                diameter_mm=d_main,
                length_km=length - prev_length,
                roughness_mm=roughness_mm,
            )
            prev_length = length
            pipe_index += 1
            prev_node = nodes[name]

        elif val["val_type"] == "comp":
            # Make compressor from node
            name_from = f"c_{node_index-1}"
            nodes[name_from] = bp_plc.Node(name=name_from, X=composition)
            node_index += 1

            # Make pipe to compressor
            p_name = f"pipe_{pipe_index}"
            pipes[p_name] = bp_plc.Pipe(
                name=p_name,
                from_node=prev_node,
                to_node=nodes[name_from],
                diameter_mm=d_main,
                length_km=length - prev_length,
                roughness_mm=roughness_mm,
            )
            prev_length = length
            pipe_index += 1

            # # Make compressor to node
            name_to = f"c_{node_index-1}"
            nodes[name_to] = bp_plc.Node(name=name_to, X=composition)
            node_index += 1

            # MAKE COMPRESSOR
            comp_name = f"c_{comp_index}"
            compressors[comp_name] = bp_plc.Compressor(
                name=comp_name,
                from_node=nodes[name_from],
                to_node=nodes[name_to],
                pressure_out_mpa_g=comp_p_out / gl.MPA2PA,
                original_rating_MW=0,
                fuel_extract=not new_comps_elec,
            )
            if not new_comps_elec:
                compressors[comp_name].eta_comp_s = eta_s
                compressors[comp_name].eta_driver = eta_driver
            else:
                compressors[comp_name].eta_comp_s_elec = eta_s
                compressors[comp_name].eta_driver_elec = eta_driver

            comp_index += 1
            prev_node = nodes[name_to]

    # If a compressor exists in the segment
    if seg_compressor:
        comp_orig = seg_compressor[0]
        comp_name = "segment_compressor"

        # Add node after compressor
        final_node_name = "final_node"
        nodes[final_node_name] = bp_plc.Node(name=final_node_name, X=composition)

        # Update demand to be the final node
        final_demand_node_name = f"demand_{demand_index-1}"
        prev_final_node = demands[final_demand_node_name].node
        demands[final_demand_node_name].node = nodes[final_node_name]

        # Add the compressor
        compressors[comp_name] = bp_plc.Compressor(
            name=comp_name,
            from_node=prev_final_node,
            to_node=nodes[final_node_name],
            pressure_out_mpa_g=prev_ASME_pressure / gl.MPA2PA,
            original_rating_MW=comp_orig.original_rating_MW,
            fuel_extract=not comps_elec,
        )
        compressors[comp_name].eta_comp_s = comp_orig.eta_comp_s
        compressors[comp_name].eta_comp_s_elec = comp_orig.eta_comp_s_elec
        compressors[comp_name].eta_driver = comp_orig.eta_driver
        compressors[comp_name].eta_driver_elec = comp_orig.eta_driver_elec

    addl_comp_network = BlendPATH_network(
        name="addl_comps",
        pipes=pipes,
        nodes=nodes,
        demand_nodes=demands,
        supply_nodes=supplys,
        compressors=compressors,
        composition=composition,
    )
    addl_comp_network.set_eos(eos=eos)
    addl_comp_network.set_thermo_curvefit(thermo_curvefit=thermo_curvefit)

    end_node = nodes[f"ot_{node_index-2}"]
    return addl_comp_network, end_node, l_comps, pipes[list(pipes.keys())[0]]


def get_num_compressors(
    composition: bp_plc.Composition,
    p_in: float,
    p_out: float,
    offtakes: list,
    offtakes_mdot: list,
    d: float,
    l_total: float,
    cr_max: float,
    roughness_mm: float,
    eta_s: float,
    eta_driver: float,
    new_comps_elec: bool,
    comp_p_out: float,
    seg_compressor: list,
    prev_ASME_pressure: float,
    comps_elec: bool,
    eos: bp_plc.eos._EOS_OPTIONS = "rk",
    thermo_curvefit: bool = False,
) -> tuple:
    all_mdot = offtakes_mdot
    if offtakes[-1] == 0:
        offtakes = offtakes[:-1]
    n_comps = 0

    # Guess n comps
    ctu.gas.TPX = gl.T_FIXED, ct.one_atm, composition.x_str
    mw = ctu.gas.mean_molecular_weight
    zrt = 1 * ctu.gas_constant * gl.T_FIXED
    p_val = (p_in**2 - (p_in / cr_max) ** 2) ** 0.5
    f = 0.01
    d_m = d * gl.MM2M
    area = np.pi * d_m**2 / 4
    l_seg = mw / zrt * d_m / f * (sum(all_mdot) / area / p_val) ** -2

    max_comps = 200
    n_comps = min(max([int(l_total / (l_seg / gl.KM2M)), 0]), max_comps - 1)

    sols = []
    n_comps_tried = []
    relax = 1.5

    while 0 <= n_comps < 200 and n_comps not in n_comps_tried:
        addl_comps, end_node, l_comps, pipe_in = make_compressor_network(
            n_comps=n_comps,
            composition=composition,
            p_in=p_in,
            offtakes=offtakes,
            all_mdot=all_mdot,
            d_main=d,
            l_total=l_total,
            roughness_mm=roughness_mm,
            eta_s=eta_s,
            eta_driver=eta_driver,
            eos=eos,
            seg_compressor=seg_compressor,
            prev_ASME_pressure=prev_ASME_pressure,
            comps_elec=comps_elec,
            new_comps_elec=new_comps_elec,
            thermo_curvefit=thermo_curvefit,
            comp_p_out=comp_p_out,
        )
        try:
            addl_comps.solve(relax, cr_max=cr_max)
        except ValueError as err_val:
            if err_val.args[0] in ["Negative pressure", "Pressure below threshold"]:
                # if negative pressure than increase compressors
                n_comps_tried.append(n_comps)
                n_comps += 1
            if err_val.args[0] in [f"Could not converge in {gl.MAX_ITER} iterations"]:
                # if negative pressure than increase compressors
                n_comps_tried.append(n_comps)
                n_comps += 1
                # return np.inf, [], [], [], -1, None
        except ct.CanteraError:
            relax *= 1.05
            continue
        else:
            # Check if outlet pressure satisfies target pressure
            if end_node.pressure >= p_out:
                # Check if compressor ratios are too high
                cs_ratio_check = False
                cs_fuel = []
                cs_fuel_elec = []
                for cs in addl_comps.compressors.values():
                    cs_fuel.append(cs.fuel_mdot)
                    cs_fuel_elec.append(cs.fuel_electric_W)
                    if cs.get_cr_ratio() > cr_max:
                        cs_ratio_check = True
                        break
                if cs_ratio_check:
                    n_comps_tried.append(n_comps)
                    n_comps += 1
                    continue

                sols.append(
                    (
                        n_comps,
                        l_comps,
                        cs_fuel,
                        cs_fuel_elec,
                        pipe_in.m_dot,
                        addl_comps.compressors,
                    )
                )
                n_comps_tried.append(n_comps)
                n_comps -= 1
            else:
                n_comps_tried.append(n_comps)
                n_comps += 1
    return sols[-1]
