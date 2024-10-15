import copy
import itertools
import math
from collections import namedtuple

import cantera as ct
import numpy as np
from pandas import DataFrame, ExcelWriter

import BlendPATH.costing.costing as bp_cost
import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components as bp_plc
import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.network.BlendPATH_network import BlendPATH_network, Design_params


def direct_replacement(
    network: BlendPATH_network,
    ASME_params: bp_pa.ASME_consts,
    design_option: str = "b",
    new_filename: str = "modified",
    design_params: Design_params = None,
    costing_params: bp_cost.Costing_params = None,
) -> tuple:
    """
    Modify network with direct replacement method
    """

    max_CR = design_params.max_CR
    final_outlet_pressure = design_params.final_outlet_pressure_mpa_g
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

    nw = copy.deepcopy(network)

    for cs in nw.compressors.values():
        cs.fuel_extract = cs.fuel_extract = (
            cs.fuel_extract and not design_params.existing_comp_elec
        )

    # Get size pipesegments
    n_ps = len(nw.pipe_segments)

    dr_entry = namedtuple(
        "dr_entry", ["grade", "diam_i", "th", "dn", "pressure", "schedule"]
    )
    lcot_entry = namedtuple(
        "lcot_entry",
        ["lcot", "geometry", "cr", "mat_cost", "supply_comp", "supp_p", "combo"],
    )

    cr_max_total = max(max_CR)

    seg_options = [[] for _ in range(n_ps)]

    # Loop segments
    for ps_i, ps in enumerate(nw.pipe_segments):
        # Fetch diameters 5 and larger
        dn_options, od_options = ps.get_DNs(5)

        # Loop grades
        for grade in bp_pa.get_pipe_grades():
            # Loop diameters
            for dn_i, dn in enumerate(dn_options):
                # Get all schedules for this combination
                (th_valid, schedule_valid, pressure_valid) = ps.get_viable_schedules(
                    design_option=design_option,
                    ASME_params=ASME_params,
                    grade=grade,
                    ASME_pressure_flag=False,
                    DN=dn,
                    return_all=True,
                )
                # Loop schedules
                for th_i, th in enumerate(th_valid):
                    if not schedule_valid or schedule_valid[th_i] is np.nan:
                        continue

                    d_outer_mm = od_options[dn_i]
                    d_inner_mm = d_outer_mm - 2 * th

                    seg_options[ps_i].append(
                        dr_entry(
                            grade,
                            d_inner_mm,
                            th,
                            dn,
                            pressure_valid[th_i],
                            schedule_valid[th_i],
                        )
                    )

    # Run LCOT sweep
    lcot_vals = []

    seg_options_slim = [seg_options[i] for i, v in enumerate(seg_options) if v]
    valid_options = []
    if seg_options_slim:
        valid_options = list(set.intersection(*map(set, seg_options_slim)))

    sn_orig = network.supply_nodes[list(network.supply_nodes.keys())[0]]
    orig_supply_pressure = sn_orig.pressure_mpa
    sn_orig_node = nw.supply_nodes[list(nw.supply_nodes.keys())[0]].node

    # First try to solve the original network, maybe no changes needed
    no_change = False
    try:
        nw.solve()
        for pipe in ps.pipes:
            pipe.design_violation = pipe.pressure_MPa > pipe.pressure_ASME_MPa
            if pipe.design_violation:
                raise ValueError()
        if nw.pipe_segments[-1].end_node.pressure < final_outlet_pressure * gl.MPA2PA:
            raise ValueError("Final outlet pressure")
        no_change = True
    # else reset to original values. Resolve to see if issues hydraulic issues
    except ValueError:
        pass

    # Get combinations for swapping
    combos = []
    for L in range(n_ps + 1):
        for subset in itertools.combinations(range(n_ps), L):
            combos.append(subset)

    # Get original values
    og_pipe_vals = []
    og_comps_out = []
    for ps_i, ps in enumerate(nw.pipe_segments):
        og_pipe_vals.append(
            {
                "diameter_mm": ps.pipes[0].diameter_mm,
                "thickness_mm": ps.pipes[0].thickness_mm,
                "DN": ps.pipes[0].DN,
            }
        )
        og_comps_out.append(ps.comps[0].pressure_out_mpa_g if ps.comps else -1)

    if not no_change:
        sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]
        valid_options = list(set.intersection(*map(set, seg_options_slim)))

        for i, design in enumerate(valid_options):
            for combo in combos:
                # Update
                total_length_km = 0
                for ps_i, ps in enumerate(nw.pipe_segments):
                    if ps_i in combo:
                        # Make changes
                        for pipe in ps.pipes:
                            pipe.diameter_mm = design.diam_i
                            pipe.thickness_mm = design.th
                            pipe.DN = design.dn
                            pipe.A_m2 = (pipe.diameter_mm * gl.MM2M) ** 2 / 4 * np.pi
                            total_length_km += pipe.length_km
                        if ps_i > 0:
                            if nw.pipe_segments[ps_i - 1].comps:
                                comp = nw.pipe_segments[ps_i - 1].comps[0]
                                comp.pressure_out_mpa_g = design.pressure
                    else:
                        for pipe in ps.pipes:
                            pipe.diameter_mm = og_pipe_vals[ps_i]["diameter_mm"]
                            pipe.thickness_mm = og_pipe_vals[ps_i]["thickness_mm"]
                            pipe.DN = og_pipe_vals[ps_i]["DN"]
                            pipe.A_m2 = (pipe.diameter_mm * gl.MM2M) ** 2 / 4 * np.pi
                        if ps_i > 0:
                            if nw.pipe_segments[ps_i - 1].comps:
                                comp = nw.pipe_segments[ps_i - 1].comps[0]
                                comp.pressure_out_mpa_g = pipe.pressure_ASME_MPa

                inlet_pressure = (
                    design.pressure
                    if 0 in combo
                    else nw.pipe_segments[0].pressure_ASME_MPa
                )
                supp_p_list = [inlet_pressure]
                if orig_supply_pressure < inlet_pressure:
                    supp_p_list = (
                        [orig_supply_pressure]
                        + list(
                            range(
                                math.ceil(orig_supply_pressure),
                                math.floor(inlet_pressure),
                                1,
                            )
                        )
                        + [inlet_pressure]
                    )

                for supp_p in supp_p_list:
                    if supp_p > orig_supply_pressure:
                        comp_to_node = sn.node
                        new_supply_node = bp_plc.Node(
                            name="new_supply_node_comp",
                            X=sn.node.X,
                            pressure=orig_supply_pressure * gl.MPA2PA,
                        )
                        supply_comp = bp_plc.Compressor(
                            name="Supply compressor",
                            from_node=new_supply_node,
                            to_node=comp_to_node,
                            pressure_out_mpa_g=supp_p,
                            fuel_extract=not design_params.new_comp_elec,
                        )
                        sn.node = new_supply_node
                        nw.compressors["Supply compressor"] = supply_comp
                        nw.nodes["new_supply_node_comp"] = new_supply_node
                        nw.assign_node_indices()
                        nw.assign_ignore_nodes()
                        nw.assign_connections()
                    sn.pressure_mpa = min(orig_supply_pressure, supp_p)

                    try:
                        nw.solve()
                        all_fuel_MW = 0
                        elec_kW = 0
                        comp_capex = []
                        revamped_comp_capex = []
                        supply_comp_capex = 0
                        supply_comp_fuel = {"gas": 0, "elec": 0}
                        max_cr = []
                        for comp in nw.compressors.values():
                            # Check compression ratio
                            this_cr = comp.get_cr_ratio()
                            if (this_cr > cr_max_total) and (
                                comp.name != "Supply compressor"
                            ):
                                raise ValueError("Compression ratio")
                            max_cr.append(comp.get_cr_ratio())

                            # Get CAPEX of compressors
                            if comp.name == "Supply compressor":
                                supply_comp_capex += comp.get_cap_cost(
                                    cp=costing_params,
                                    to_electric=design_params.existing_comp_elec,
                                )
                                supply_comp_fuel = {
                                    "gas": comp.fuel_w
                                    * gl.W2MW
                                    * gl.MW2MMBTUDAY
                                    / nw.capacity_MMBTU_day,
                                    "elec": comp.fuel_electric_W
                                    / gl.KW2W
                                    / nw.capacity_MMBTU_day
                                    * gl.DAY2HR,
                                }
                                continue

                            # Add up fuel costs
                            all_fuel_MW += comp.fuel_w * gl.W2MW
                            elec_kW += comp.fuel_electric_W / gl.KW2W

                            comp_capex.append(
                                comp.get_cap_cost(
                                    cp=costing_params,
                                    to_electric=design_params.existing_comp_elec,
                                )
                            )
                            revamped_comp_capex.append(
                                comp.get_cap_cost(
                                    cp=costing_params,
                                    revamp=True,
                                    to_electric=design_params.existing_comp_elec,
                                )
                            )

                        if (
                            nw.pipe_segments[-1].end_node.pressure
                            < final_outlet_pressure * gl.MPA2PA
                        ):
                            raise ValueError("Final outlet pressure")

                        d_inner_mm = design.diam_i
                        d_outer_mm = d_inner_mm + 2 * design.th

                        cost = bp_cost.get_pipe_material_cost(
                            cp=costing_params,
                            di_mm=d_inner_mm,
                            do_mm=d_outer_mm,
                            l_km=total_length_km,
                            grade=design.grade,
                        )

                        anl_cap = bp_cost.get_pipe_other_cost(
                            cp=costing_params,
                            d_mm=design.dn,
                            l_km=total_length_km,
                            anl_types=["Labor", "Misc"],
                        )

                        new_pipe_cap = cost + sum(anl_cap.values())

                        all_pipes_len = []
                        pipe_dns_lens = {}
                        for pipe in nw.pipes.values():
                            if pipe.DN in pipe_dns_lens.keys():
                                pipe_dns_lens[pipe.DN] += pipe.length_km
                            else:
                                pipe_dns_lens[pipe.DN] = pipe.length_km
                        all_pipes_len = [
                            (dn, length) for dn, length in pipe_dns_lens.items()
                        ]

                        # Get all demands
                        demands_MW = [
                            demand.flowrate_MW for demand in nw.demand_nodes.values()
                        ]

                        # Get meter,ILI,valve costs
                        meter_cost = bp_cost.meter_reg_station_cost(
                            costing_params, demands_MW
                        )
                        ili_costs = bp_cost.ili_cost(costing_params, all_pipes_len)
                        valve_cost = bp_cost.valve_replacement_cost(
                            costing_params, all_pipes_len, ASME_params.location_class
                        )

                        price_breakdown = bp_cost.calc_lcot(
                            json_file=costing_params.casestudy_name,
                            capacity=nw.capacity_MMBTU_day,
                            new_pipe_cap=new_pipe_cap,
                            comp_cost=comp_capex,
                            revamped_comp_capex=revamped_comp_capex,
                            supply_comp_capex=supply_comp_capex,
                            compressor_fuel=all_fuel_MW
                            * gl.MW2MMBTUDAY
                            / nw.capacity_MMBTU_day,
                            compressor_fuel_elec=elec_kW
                            / nw.capacity_MMBTU_day
                            * gl.DAY2HR,
                            supply_comp_fuel=supply_comp_fuel,
                            cs_cost=costing_params.cf_price,
                            elec_cost=costing_params.elec_price,
                            meter_cost=meter_cost,
                            ili_costs=ili_costs,
                            valve_cost=valve_cost,
                            original_network_residual_value=costing_params.original_pipeline_cost,
                            financial_overrides=costing_params.financial_overrides,
                        )

                        lcot = price_breakdown["LCOT: Levelized cost of transport"]
                        lcot_vals.append(
                            lcot_entry(
                                lcot,
                                design,
                                max(max_cr) if max_cr else np.nan,
                                cost,
                                "Supply compressor" in nw.compressors.keys(),
                                supp_p=supp_p,
                                combo=combo,
                            )
                        )
                    except (ValueError, ct.CanteraError):
                        pass
                    finally:
                        # reset supply node and compressor if needed
                        if "Supply compressor" in nw.compressors.keys():
                            sn_node = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]
                            sn_node.node = sn_orig_node
                            sn_node.pressure = network.supply_nodes[
                                list(network.supply_nodes.keys())[0]
                            ].pressure_mpa
                            nw.compressors.pop("Supply compressor")
                            nw.nodes.pop("new_supply_node_comp")
                            nw.assign_node_indices()
                            nw.assign_ignore_nodes()
                            nw.assign_connections()

        ######################################

    add_supply_comp = False
    if valid_options and not no_change:
        lcot_list = [x.lcot for x in lcot_vals]
        min_lcot_ind = lcot_list.index(min(lcot_list))
        min_geom = lcot_vals[min_lcot_ind].geometry
        add_supply_comp = lcot_vals[min_lcot_ind].supply_comp
        supp_p = lcot_vals[min_lcot_ind].supp_p
        combo_final = lcot_vals[min_lcot_ind].combo

    res = {
        x: {
            v: []
            for v in [
                "grades",
                "costs",
                "ths",
                "schedules",
                "pressures",
                "inner diameters",
                "DN",
                "lengths",
                "ps",
                "name",
            ]
        }
        for x in nw.pipes.keys()
    }
    if not no_change:
        for ps_i, ps in enumerate(network.pipe_segments):
            if ps_i in combo_final:
                for pipe in ps.pipes:
                    if (
                        (min_geom.dn == pipe.DN)
                        and (min_geom.schedule == pipe.schedule)
                        and (min_geom.grade == pipe.grade)
                    ):
                        continue
                    res[pipe.name]["grades"].append(min_geom.grade)
                    res[pipe.name]["costs"].append(lcot_vals[min_lcot_ind].mat_cost)
                    res[pipe.name]["ths"].append(min_geom.th)
                    res[pipe.name]["schedules"].append(min_geom.schedule)
                    res[pipe.name]["pressures"].append(min_geom.pressure)
                    res[pipe.name]["inner diameters"].append(min_geom.diam_i)
                    res[pipe.name]["DN"].append(min_geom.dn)
                    res[pipe.name]["lengths"].append(pipe.length_km)
                    res[pipe.name]["ps"].append(ps_i)
                    res[pipe.name]["name"].append(pipe.name)

    # Can combine this with previous step
    # Get the minimum cost option per pipe and related geometry
    min_cost = {}
    combined_pipe = {
        x: []
        for x in [
            "D_S_G",
            "DN",
            "sch",
            "grade",
            "length",
            "mat_cost",
            "anl_costs",
            "total cost",
        ]
    }
    for pipe, val in res.items():
        if not val["costs"]:
            continue
        mincost_i = min(val["costs"])
        mincost_ind = val["costs"].index(mincost_i)
        min_cost[pipe] = {x: v[mincost_ind] for x, v in val.items()}

        # Combine by DN,sch,grade
        combined_ind = f"{min_cost[pipe]['DN']};;{min_cost[pipe]['schedules']};;{min_cost[pipe]['grades']}"
        min_cost[pipe]["D_S_G"] = combined_ind
        # If the same DN, sch, and grade, combo exists, then add to the length and cost
        # Otherwise make a new entry
        if combined_ind in combined_pipe["D_S_G"]:
            dsg_ind = combined_pipe["D_S_G"].index(combined_ind)
            combined_pipe["length"][dsg_ind] += min_cost[pipe]["lengths"]
            combined_pipe["mat_cost"][dsg_ind] += min_cost[pipe]["costs"]
        else:
            combined_pipe["DN"].append(min_cost[pipe]["DN"])
            combined_pipe["sch"].append(min_cost[pipe]["schedules"])
            combined_pipe["grade"].append(min_cost[pipe]["grades"])
            combined_pipe["D_S_G"].append(combined_ind)
            combined_pipe["length"].append(min_cost[pipe]["lengths"])
            combined_pipe["mat_cost"].append(min_cost[pipe]["costs"])
    # Temporary fix for setting total material cost
    if combined_pipe["mat_cost"]:
        combined_pipe["mat_cost"][-1] = lcot_vals[min_lcot_ind].mat_cost

    # Remake file

    # Make new pipes sheet
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
    for pipe in network.pipes.values():

        # These values don't change from DR method
        new_pipes["pipe_name"].append(pipe.name)
        new_pipes["from_node"].append(pipe.from_node.name)
        new_pipes["to_node"].append(pipe.to_node.name)
        new_pipes["length_km"].append(pipe.length_km)
        new_pipes["roughness_mm"].append(pipe.roughness_mm)

        # These values update
        repd = pipe.name in min_cost.keys()
        inner_diam = (
            min_cost[pipe.name]["inner diameters"] if repd else pipe.diameter_mm
        )
        thickness = min_cost[pipe.name]["ths"] if repd else pipe.thickness_mm
        grade = min_cost[pipe.name]["grades"] if repd else pipe.grade
        new_pipes["diameter_mm"].append(inner_diam)
        new_pipes["thickness_mm"].append(thickness)
        new_pipes["steel_grade"].append(grade)

    # Make new nodes sheet
    col_names = ["node_name", "p_max_mpa_g"]
    new_nodes = {x: [] for x in col_names}
    for node in network.nodes.values():
        new_nodes["node_name"].append(node.name)
        p_max = np.inf
        for pipe in node.connections["Pipe"]:
            min_cost_val = (
                min_cost[pipe.name]["pressures"]
                if pipe.name in min_cost.keys()
                else pipe.pressure_ASME_MPa
            )
            p_max = min(p_max, min_cost_val)
        new_nodes["p_max_mpa_g"].append(p_max)

    # Make new compressors
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

        # Assume to node has to be the outlet pressure
        for pipe in comp.to_node.connections["Pipe"]:
            for ps_i, ps in enumerate(nw.pipe_segments):
                if pipe in ps.pipes:
                    if ps_i in combo_final:
                        if ps_i > 0 and nw.pipe_segments[ps_i - 1].comps:
                            p_max = min_geom.pressure
                    else:
                        p_max = ps.pressure_ASME_MPa

        # p_max = np.inf
        # for pipe in comp.to_node.connections["Pipe"]:
        #     min_cost_val = (
        #         min_cost[pipe.name]["pressures"]
        #         if pipe.name in min_cost.keys()
        #         else (
        #             min_geom.pressure if not no_change else pipe.pipe.pressure_ASME_MPa
        #         )
        #     )
        #     p_max = min(p_max, min_cost_val)

        new_comps["pressure_out_mpa_g"].append(p_max)

    # Make new supply
    col_names = ["supply_name", "node_name", "pressure_mpa_g", "flowrate_MW"]
    new_supply = {x: [] for x in col_names}
    for supply in network.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)

        # Assume to node has to be the outlet pressure
        # p_max = np.inf
        # for pipe in supply.node.connections["Pipe"]:
        #     min_cost_val = (
        #         min_cost[pipe.name]["pressures"]
        #         if pipe.name in min_cost.keys()
        #         else pipe.pressure_ASME_MPa
        #     )
        #     p_max = min(p_max, min_cost_val)

        new_supply["pressure_mpa_g"].append(
            supp_p if not no_change else supply.pressure_mpa
        )
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
        for pipe in sn.node.connections["Pipe"]:
            min_cost_val = (
                min_cost[pipe.name]["pressures"]
                if pipe.name in min_cost.keys()
                else pipe.pressure_ASME_MPa
            )
            p_max = min(p_max, min_cost_val)
        new_comps["pressure_out_mpa_g"].insert(0, supp_p)

        # Add new node
        new_nodes["node_name"].insert(0, "Supply compressor from_node")
        new_nodes["p_max_mpa_g"].insert(0, p_max)

        # Update supply node
        new_supply["node_name"][-1] = "Supply compressor from_node"
        new_supply["pressure_mpa_g"][-1] = supply.pressure_mpa

    # Make new demands
    col_names = ["demand_name", "node_name", "flowrate_MW"]
    new_demand = {x: [] for x in col_names}
    for demand in network.demand_nodes.values():
        new_demand["demand_name"].append(demand.name)
        new_demand["node_name"].append(demand.node.name)
        new_demand["flowrate_MW"].append(demand.flowrate_MW)

    # Make new composition
    col_names = ["SPECIES", "X"]
    new_composition = {x: [] for x in col_names}
    for species, x in network.composition.x.items():
        new_composition["SPECIES"].append(species)
        new_composition["X"].append(x)

    new_pipes = DataFrame(new_pipes)
    new_nodes = DataFrame(new_nodes)
    new_comps = DataFrame(new_comps)
    new_supply = DataFrame(new_supply)
    new_demand = DataFrame(new_demand)
    new_composition = DataFrame(new_composition)

    with ExcelWriter(new_filename) as writer:
        new_pipes.to_excel(writer, sheet_name="PIPES", index=False)
        new_nodes.to_excel(writer, sheet_name="NODES", index=False)
        new_comps.to_excel(writer, sheet_name="COMPRESSORS", index=False)
        new_supply.to_excel(writer, sheet_name="SUPPLY", index=False)
        new_demand.to_excel(writer, sheet_name="DEMAND", index=False)
        new_composition.to_excel(writer, sheet_name="COMPOSITION", index=False)

    return min_cost, combined_pipe
