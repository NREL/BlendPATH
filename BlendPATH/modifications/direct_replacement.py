import copy
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
        "lcot_entry", ["lcot", "geometry", "cr", "mat_cost", "supply_comp"]
    )

    cr_max_total = max(max_CR)

    seg_options = [[]] * n_ps
    diameter_sweep = [False] * n_ps
    seg_no_change = [False] * n_ps

    # Loop thru segments and make replacement options
    for ps_i, ps in reversed(list(enumerate(nw.pipe_segments))):
        # Try changing supply pressure to ASME pressure, see if hydraulics solve
        try:
            supply_orig = ps.start_node.pressure
            ps.start_node.pressure = ps.pipes[0].pressure_ASME_MPa * gl.MPA2PA
            for sn in nw.supply_nodes.values():
                if ps.start_node is sn.node:
                    sn.pressure_mpa = ps.pipes[0].pressure_ASME_MPa
                    break
            comp_orig = [comp.pressure_out_mpa_g for comp in nw.compressors.values()]
            for comp in nw.compressors.values():
                comp.pressure_out_mpa_g = comp.to_node.connections["Pipe"][
                    0
                ].pressure_ASME_MPa

            nw.solve()
            for pipe in ps.pipes:
                pipe.design_violation = pipe.pressure_MPa > pipe.pressure_ASME_MPa
                if pipe.design_violation:
                    raise ValueError()
            if (
                nw.pipe_segments[-1].end_node.pressure
                < final_outlet_pressure * gl.MPA2PA
            ):
                raise ValueError("Final outlet pressure")
        # else reset to original values. Resolve to see if issues hydraulic issues
        except ValueError:
            ps.start_node.pressure = supply_orig
            for sn in nw.supply_nodes.values():
                if ps.start_node is sn.node:
                    sn.pressure_mpa = supply_orig / gl.MPA2PA
                    break
            for comp_i, comp in enumerate(nw.compressors.values()):
                comp.pressure_out_mpa_g = comp_orig[comp_i]
            try:
                nw.solve()
                for pipe in ps.pipes:
                    pipe.design_violation = pipe.pressure_MPa > pipe.pressure_ASME_MPa
                if (
                    nw.pipe_segments[-1].end_node.pressure
                    < final_outlet_pressure * gl.MPA2PA
                ):
                    raise ValueError("Final outlet pressure")
            except ValueError:
                diameter_sweep[ps_i] = True

        # Check if compressor ratio is over max
        if ps.comps:
            ps_comp = ps.comps[0]
            if ps_comp.get_cr_ratio() > cr_max_total:
                diameter_sweep[ps_i] = True

        # If no diameter sweep required, theck check pressure violations
        if not diameter_sweep[ps_i]:
            ps.check_p_violations()
            # If no pressure violations, still check other diameters for comparison. Else diameter sweep
            if not ps.pressure_violation:
                seg_no_change[ps_i] = True
                continue
            else:
                diameter_sweep[ps_i] = True

        dn_options, od_options = (
            ps.get_DNs(5) if diameter_sweep[ps_i] else ps.get_DNs(1)
        )

        seg_options[ps_i] = []

        # Loop grades
        for grade in bp_pa.get_pipe_grades():
            # Loop valid diameters
            for dn_i, dn in enumerate(dn_options):
                sch_flag = diameter_sweep[ps_i]
                (th_valid, schedule_valid, pressure_valid) = ps.get_viable_schedules(
                    design_option=design_option,
                    ASME_params=ASME_params,
                    grade=grade,
                    ASME_pressure_flag=sch_flag,
                    DN=dn if sch_flag else None,
                    return_all=True,
                )
                # Loop thicknesses
                for th_i, th in enumerate(th_valid):
                    # If invalid, then go to next thickness
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
    rerun_for_all = True
    while rerun_for_all:

        seg_options_slim = [seg_options[i] for i, v in enumerate(seg_options) if v]
        valid_options = []
        if seg_options_slim:
            valid_options = list(set.intersection(*map(set, seg_options_slim)))

        total_length_km = sum(
            [
                ps.length_km if not seg_no_change[ps_i] else 0
                for ps_i, ps in enumerate(nw.pipe_segments)
            ]
        )
        orig_supply_pressure = network.supply_nodes[
            list(network.supply_nodes.keys())[0]
        ].pressure_mpa
        for i, design in enumerate(valid_options):

            sn_orig_node = nw.supply_nodes[list(nw.supply_nodes.keys())[0]].node
            for ps_i in range(n_ps):
                if seg_no_change[ps_i]:
                    continue
                ps = nw.pipe_segments[ps_i]
                ps_nodes = [x.name for x in ps.nodes]

                # Update pipe diameters
                for pipe in ps.pipes:
                    pipe.diameter_mm = design.diam_i
                    pipe.thickness_mm = design.th
                    pipe.DN = design.dn
                    pipe.A_m2 = (pipe.diameter_mm * gl.MM2M) ** 2 / 4 * np.pi

                # Update supply node if needed
                sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]
                # for sn in nw.supply_nodes.values():
                if sn.node.name in ps_nodes:
                    sn.pressure_mpa = min(design.pressure, orig_supply_pressure)
                    # Check if compressor needs to be added at supply

                    if design.pressure > orig_supply_pressure:
                        sn.pressure_mpa = orig_supply_pressure
                        comp_to_node = sn.node
                        new_supply_node = bp_plc.Node(
                            name="new_supply_node_comp",
                            X=sn.node.X,
                            pressure=orig_supply_pressure * gl.MPA2PA,
                        )
                        supply_comp = bp_plc.Compressor(
                            name="new_supply_comp",
                            from_node=new_supply_node,
                            to_node=comp_to_node,
                            pressure_out_mpa_g=design.pressure,
                            fuel_extract=not design_params.new_comp_elec,
                        )
                        sn.node = new_supply_node
                        nw.compressors["new_supply_comp"] = supply_comp
                        nw.nodes["new_supply_node_comp"] = new_supply_node
                        nw.assign_node_indices()
                        nw.assign_ignore_nodes()
                        nw.assign_connections()
                        pass
                # Update compressor outlet if needed (assumes pipe segments in order)
                if ps_i > 0:
                    if nw.pipe_segments[ps_i - 1].comps:
                        comp = nw.pipe_segments[ps_i - 1].comps[0]
                        comp.pressure_out_mpa_g = design.pressure

            try:  # try to solve new network, check compression ratios and pressures
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
                    if (this_cr > cr_max_total) and (comp.name != "new_supply_comp"):
                        raise ValueError("Compression ratio")
                    max_cr.append(comp.get_cr_ratio())

                    # Get CAPEX of compressors
                    if comp.name == "new_supply_comp":
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
                all_pipes_len = [(dn, length) for dn, length in pipe_dns_lens.items()]

                # Get all demands
                demands_MW = [demand.flowrate_MW for demand in nw.demand_nodes.values()]

                # Get meter,ILI,valve costs
                meter_cost = bp_cost.meter_reg_station_cost(costing_params, demands_MW)
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
                    compressor_fuel_elec=elec_kW / nw.capacity_MMBTU_day * gl.DAY2HR,
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
                        "new_supply_comp" in nw.compressors.keys(),
                    )
                )

            except ValueError:
                continue
            except ct.CanteraError:
                continue
            finally:
                # reset supply node and compressor if needed
                if "new_supply_comp" in nw.compressors.keys():
                    sn_node = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]
                    sn_node.node = sn_orig_node
                    sn_node.pressure = network.supply_nodes[
                        list(network.supply_nodes.keys())[0]
                    ].pressure_mpa
                    nw.compressors.pop("new_supply_comp")
                    nw.nodes.pop("new_supply_node_comp")
                    nw.assign_node_indices()
                    nw.assign_ignore_nodes()
                    nw.assign_connections()

        # Change to allow all segments to change if
        if not lcot_vals and valid_options:
            rerun_for_all = True
            seg_no_change = [False] * n_ps
        else:
            rerun_for_all = False

    add_supply_comp = False
    if valid_options:
        lcot_list = [x.lcot for x in lcot_vals]
        min_lcot_ind = lcot_list.index(min(lcot_list))
        min_geom = lcot_vals[min_lcot_ind].geometry
        add_supply_comp = lcot_vals[min_lcot_ind].supply_comp

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
    for ps_i, ps in enumerate(nw.pipe_segments):
        if seg_no_change[ps_i]:
            continue
        for pipe in ps.pipes:
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
    pass

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
            comp.eta_driver if comp.fuel_extract else comp.eta_driver_elec_used
        )

        # Assume to node has to be the outlet pressure
        p_max = np.inf
        for pipe in comp.to_node.connections["Pipe"]:
            min_cost_val = (
                min_cost[pipe.name]["pressures"]
                if pipe.name in min_cost.keys()
                else pipe.pressure_ASME_MPa
            )
            p_max = min(p_max, min_cost_val)

        new_comps["pressure_out_mpa_g"].append(p_max)

    # Make new supply
    col_names = ["supply_name", "node_name", "pressure_mpa_g", "flowrate_MW"]
    new_supply = {x: [] for x in col_names}
    for supply in network.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)

        # Assume to node has to be the outlet pressure
        p_max = np.inf
        for pipe in supply.node.connections["Pipe"]:
            min_cost_val = (
                min_cost[pipe.name]["pressures"]
                if pipe.name in min_cost.keys()
                else pipe.pressure_ASME_MPa
            )
            p_max = min(p_max, min_cost_val)

        new_supply["pressure_mpa_g"].append(p_max)
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
        new_comps["pressure_out_mpa_g"].insert(0, p_max)

        # Add new node
        new_nodes["node_name"].insert(0, "Supply compressor from_node")
        new_nodes["p_max_mpa_g"].insert(0, p_max)

        # Update supply node
        new_supply["node_name"][-1] = "Supply compressor from_node"
        new_supply["pressure_mpa_g"][-1] = sn.pressure_mpa

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
