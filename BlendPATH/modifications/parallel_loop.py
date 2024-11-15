import copy
import math

import numpy as np
from pandas import DataFrame, ExcelWriter

import BlendPATH.costing.costing as bp_cost
import BlendPATH.Global as gl
import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.network import pipeline_components as bp_plc
from BlendPATH.network.BlendPATH_network import BlendPATH_network, Design_params


def parallel_loop(
    network: BlendPATH_network,
    ASME_params: bp_pa.ASME_consts,
    design_params: Design_params = None,
    design_option: str = "b",
    new_filename: str = "modified",
    costing_params: bp_cost.Costing_params = None,
) -> tuple:
    """
    Modify the pipeline with parallel looping
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

    # For new compressors (supply compressor in PL case)
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

    # Initialize results. This is per CR and per pipe segment
    res = [
        {
            i: {
                v: []
                for v in [
                    "costs",
                    "mat_costs",
                    "grades",
                    "ths",
                    "schedules",
                    "pressures",
                    "inner diameters",
                    "DN",
                    "loop_length",
                    "add_supply_comp",
                    "inlet_p",
                ]
            }
            for i, _ in enumerate(nw.pipe_segments)
        }
        for _ in range(n_cr)
    ]

    # Loop thru compression ratios
    for cr_i, CR_ratio in enumerate(max_CR):

        # Loop thru pipe segments
        prev_ASME_pressure = -1
        m_dot_in_prev = nw.pipe_segments[-1].mdot_out
        for ps_i, ps in reversed(list(enumerate(nw.pipe_segments))):
            # Take the next 10 greater than or equal DNs
            dn_options, od_options = ps.get_DNs(10)

            # Setup the list to loop thru supply pressures
            supp_p_list = [ps.pressure_ASME_MPa]
            # Only relevant for first segment
            if ps_i == 0:
                og_pressure = nw.supply_nodes[
                    list(nw.supply_nodes.keys())[0]
                ].pressure_mpa

                # This takes integer values between the original
                # pressure and the MAOP
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
            if ps_i == n_ps - 1:
                pressure_out_MPa = final_outlet_pressure
            pressure_out_Pa = pressure_out_MPa * gl.MPA2PA
            l_total = ps.length_km
            ps_nodes = [x.name for x in ps.nodes]

            supply_comp_capex = 0
            supply_comp_fuel = {"gas": 0, "elec": 0}

            # Loop through pipe steel grades
            for grade in bp_pa.get_pipe_grades():
                # Speed up calculations: lowest DN is cheapest so skip higher DNs
                dn_satisfied = False

                # Loop thru diameters >= current DN
                for dn_i, dn in enumerate(dn_options):
                    # Skip larger DN's if smaller solution found
                    if dn_satisfied:
                        break

                    # Get closest schedule that satifies design pressure
                    (th, schedule, pressure) = ps.get_viable_schedules(
                        design_option=design_option,
                        ASME_params=ASME_params,
                        grade=grade,
                        ASME_pressure_flag=True,
                        DN=dn,
                    )
                    # Skip if no viable schedules with this grade
                    if schedule is np.nan:
                        continue

                    # Get inner and outer diameters
                    d_outer_mm = od_options[dn_i]
                    d_inner_mm = d_outer_mm - 2 * th

                    # Calculate all offtakes -- adds the pipe segment outlet as a
                    # offtake if it is not already
                    all_mdot = ps.offtake_mdots.copy()
                    if (
                        len(ps.offtake_mdots) == 0
                        or abs(ps.offtake_mdots[-1] - m_dot_in_prev)
                        / ps.offtake_mdots[-1]
                        > 0.01
                    ):
                        all_mdot.append(m_dot_in_prev)
                    else:
                        all_mdot[-1] = m_dot_in_prev

                    # Loop through supply pressures
                    supp_p_min_res = []
                    for sup_p in supp_p_list:

                        loop_length, m_dot_seg, comp_out = get_loop_length(
                            composition=composition,
                            d_main=ps.diameter,
                            d_loop=d_inner_mm,
                            l_total=l_total,
                            p_in=sup_p,
                            p_out_target=pressure_out_Pa,
                            offtakes=ps.offtake_lengths,
                            offtakes_mdot=all_mdot,
                            roughness_mm=ps.pipes[0].roughness_mm,
                            eos=nw.eos,
                            thermo_curvefit=nw.thermo_curvefit,
                            seg_compressor=ps.comps,
                            costing_params=costing_params,
                            design_params=design_params,
                            prev_ASME_pressure=prev_ASME_pressure,
                        )

                        if np.isnan(loop_length):
                            continue
                        dn_satisfied = True

                        # Get material cost
                        loop_cost = bp_cost.get_pipe_material_cost(
                            cp=costing_params,
                            di_mm=d_inner_mm,
                            do_mm=d_outer_mm,
                            l_km=loop_length,
                            grade=grade,
                        )

                        # Get other pipe costs
                        new_pipe_cap = 0
                        if loop_length > 0:
                            anl_cap = bp_cost.get_pipe_other_cost(
                                cp=costing_params,
                                d_mm=dn,
                                l_km=loop_length,
                                anl_types=["Labor", "Misc", "ROW"],
                            )
                            anl_cap_sum = sum(anl_cap.values())
                            new_pipe_cap = anl_cap_sum + loop_cost

                        # Unit change for segment capacity and compressor fuel
                        capacity = sum(all_mdot) * ps.HHV * gl.MW2MMBTUDAY
                        fuel_use = comp_out["fuel"] * ps.HHV * gl.MW2MMBTUDAY
                        elec_use = comp_out["elec"] * gl.DAY2HR / gl.KW2W

                        # Check if supply compressor needed
                        sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]
                        orig_supply_pressure = min(
                            sn.pressure_mpa, ps.pressure_ASME_MPa
                        )
                        add_supply_comp = False
                        if sn.node.name in ps_nodes and orig_supply_pressure < sup_p:
                            add_supply_comp = True
                            from_node = bp_plc.Node(
                                name="comp_to_node",
                                X=sn.node.X,
                                pressure=orig_supply_pressure * gl.MPA2PA,
                            )
                            supply_comp = bp_plc.Compressor(
                                name="new_supply_comp",
                                from_node=from_node,
                                to_node=bp_plc.Node(name="comp_to_node", X=sn.node.X),
                                pressure_out_mpa_g=pressure,
                                fuel_extract=not design_params.new_comp_elec,
                            )
                            supply_comp_fuel = {
                                "gas": supply_comp.get_fuel_use(m_dot=m_dot_seg)
                                * ps.HHV
                                * gl.MW2MMBTUDAY
                                / capacity,
                                "elec": supply_comp.fuel_electric_W
                                * gl.DAY2HR
                                / gl.KW2W
                                / capacity,
                            }

                            supply_comp_capex += supply_comp.get_cap_cost(
                                cp=costing_params
                            )

                        # Get meter,ILI,valve costs
                        meter_cost = bp_cost.meter_reg_station_cost(
                            cp=costing_params, demands_MW=demands_MW
                        )
                        ili_costs = bp_cost.ili_cost(
                            cp=costing_params, pipe_added=[(dn, loop_length)]
                        )
                        valve_cost = bp_cost.valve_replacement_cost(
                            costing_params,
                            [(dn, loop_length)],
                            ASME_params.location_class,
                        )

                        price_breakdown = bp_cost.calc_lcot(
                            json_file=costing_params.casestudy_name,
                            capacity=capacity,
                            new_pipe_cap=new_pipe_cap,
                            comp_cost=[comp_out["cost"]],
                            revamped_comp_capex=[comp_out["revamp"]],
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

                        # Add as an entry to determine best supply pressure
                        supp_p_min_res.append(
                            (
                                segment_lcot,
                                loop_cost,
                                loop_length,
                                add_supply_comp,
                                sup_p,
                            )
                        )

                    # Choose best supply pressure
                    if not supp_p_min_res:
                        continue
                    lcot_p_spply = [x[0] for x in supp_p_min_res]
                    idxmin_lcot_p_supp = lcot_p_spply.index(min(lcot_p_spply))

                    # Add to results
                    res[cr_i][ps_i]["costs"].append(
                        supp_p_min_res[idxmin_lcot_p_supp][0]
                    )
                    res[cr_i][ps_i]["mat_costs"].append(
                        supp_p_min_res[idxmin_lcot_p_supp][1]
                    )
                    res[cr_i][ps_i]["grades"].append(grade)
                    res[cr_i][ps_i]["ths"].append(th)
                    res[cr_i][ps_i]["schedules"].append(schedule)
                    res[cr_i][ps_i]["pressures"].append(pressure)
                    res[cr_i][ps_i]["inner diameters"].append(d_inner_mm)
                    res[cr_i][ps_i]["DN"].append(dn)
                    res[cr_i][ps_i]["loop_length"].append(
                        supp_p_min_res[idxmin_lcot_p_supp][2]
                    )
                    res[cr_i][ps_i]["add_supply_comp"].append(
                        supp_p_min_res[idxmin_lcot_p_supp][3]
                    )
                    res[cr_i][ps_i]["inlet_p"].append(
                        supp_p_min_res[idxmin_lcot_p_supp][4]
                    )

            # Update the previous segment pressure for use in fuel extraction calc
            # m_dot_seg isnt changing with grade and diameter, since pipe is
            # rated to original pipe MAOP
            prev_ASME_pressure = pressure_in_Pa
            m_dot_in_prev = m_dot_seg

    # Choose lowest LCOT solution based on for each CR
    cr_lcot_sum = [0] * n_cr
    for cr_i in range(n_cr):
        cr_lcot_sum[cr_i] = sum(
            [
                min(res[cr_i][x]["costs"]) if res[cr_i][x]["costs"] else np.inf
                for x in range(n_ps)
            ]
        )
    cr_min_index = cr_lcot_sum.index(min(cr_lcot_sum))
    res = res[cr_min_index]

    ######### GET MINIMUM CASE
    min_vals = {}
    add_supply_comp_min = False
    for ps in res:
        min_index = np.argmin(res[ps]["costs"])
        min_vals[ps] = {x: res[ps][x][min_index] for x in res[ps_i].keys()}
        if min_vals[ps]["add_supply_comp"]:
            add_supply_comp_min = True
    pass

    ###### REMAKE PL FILE

    col_names = ["node_name", "p_max_mpa_g"]
    new_nodes = {x: [] for x in col_names}

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
    added_pipe_names = [""] * n_ps

    for ps_i in min_vals:
        loop_grade = min_vals[ps_i]["grades"]
        loop_th = min_vals[ps_i]["ths"]
        loop_diam = min_vals[ps_i]["inner diameters"]
        loop_length = min_vals[ps_i]["loop_length"]

        # get pipe segment object
        ps = nw.pipe_segments[ps_i]
        p_max_seg = ps.pressure_ASME_MPa

        # Get total loop length
        l_total = nw.pipe_segments[ps_i].length_km

        # loop thru all nodes in pipe_segment, update pressure
        for node in nw.pipe_segments[ps_i].nodes:
            new_nodes["node_name"].append(node.name)
            new_nodes["p_max_mpa_g"].append(p_max_seg)

        # Add pipes until past the offtake length
        l_added = 0
        loop_done = False
        if loop_length == 0 or loop_length == l_total:
            loop_done = True

        for pipe in ps.pipes:
            pipe_l = pipe.length_km
            if not loop_done and l_added + pipe_l > loop_length:
                # add segmented pipe x2, intermediate node, loop pipe
                loop_cxn_name = f"loop_cxn_node_ps_{ps_i}"

                # Get index of node before loop cxn
                insert_ind = new_nodes["node_name"].index(pipe.to_node.name)
                # Add new node for loop connection
                new_nodes["node_name"].insert(insert_ind, loop_cxn_name)
                new_nodes["p_max_mpa_g"].insert(insert_ind, p_max_seg)

                # Add segmented pipe prior to loop cxn
                pre_pipe_len = loop_length - l_added
                new_pipes["pipe_name"].append(f"{pipe.name}_pre_loop_cxn")
                new_pipes["from_node"].append(pipe.from_node.name)
                new_pipes["to_node"].append(loop_cxn_name)
                new_pipes["length_km"].append(pre_pipe_len)
                new_pipes["roughness_mm"].append(pipe.roughness_mm)
                new_pipes["diameter_mm"].append(pipe.diameter_mm)
                new_pipes["thickness_mm"].append(pipe.thickness_mm)
                new_pipes["steel_grade"].append(pipe.grade)

                # Add looped pipe to loop cxn
                added_pipe_names[ps_i] = f"PS_{ps_i}_loop"
                new_pipes["pipe_name"].append(added_pipe_names[ps_i])
                new_pipes["from_node"].append(ps.start_node.name)
                new_pipes["to_node"].append(loop_cxn_name)
                new_pipes["length_km"].append(loop_length)
                new_pipes["roughness_mm"].append(pipe.roughness_mm)
                new_pipes["diameter_mm"].append(loop_diam)
                new_pipes["thickness_mm"].append(loop_th)
                new_pipes["steel_grade"].append(loop_grade)

                # Add segmented pipe after loop cxn
                new_pipes["pipe_name"].append(f"{pipe.name}_post_loop_cxn")
                new_pipes["from_node"].append(loop_cxn_name)
                new_pipes["to_node"].append(pipe.to_node.name)
                new_pipes["length_km"].append(pipe_l - pre_pipe_len)
                new_pipes["roughness_mm"].append(pipe.roughness_mm)
                new_pipes["diameter_mm"].append(pipe.diameter_mm)
                new_pipes["thickness_mm"].append(pipe.thickness_mm)
                new_pipes["steel_grade"].append(pipe.grade)

                loop_done = True
            else:
                # add pipe as normal
                new_pipes["pipe_name"].append(pipe.name)
                new_pipes["from_node"].append(pipe.from_node.name)
                new_pipes["to_node"].append(pipe.to_node.name)
                new_pipes["length_km"].append(pipe.length_km)
                new_pipes["roughness_mm"].append(pipe.roughness_mm)
                new_pipes["diameter_mm"].append(pipe.diameter_mm)
                new_pipes["thickness_mm"].append(pipe.thickness_mm)
                new_pipes["steel_grade"].append(pipe.grade)

            # update cumulative length off pipe added
            l_added += pipe.length_km

        # If pipeline has no looping, then nothing needs to be done
        if loop_length == 0:
            pass
        # If 100% looping, add in the loop
        if loop_length == l_total:
            added_pipe_names[ps_i] = f"PS_{ps_i}_loop"
            new_pipes["pipe_name"].append(added_pipe_names[ps_i])
            new_pipes["from_node"].append(ps.start_node.name)
            new_pipes["to_node"].append(ps.end_node.name)
            new_pipes["length_km"].append(l_total)
            new_pipes["roughness_mm"].append(pipe.roughness_mm)
            new_pipes["diameter_mm"].append(loop_diam)
            new_pipes["thickness_mm"].append(loop_th)
            new_pipes["steel_grade"].append(loop_grade)

    # Add demand, supply, compressors, as usual

    # Add supply
    col_names = ["supply_name", "node_name", "pressure_mpa_g", "flowrate_MW"]
    new_supply = {x: [] for x in col_names}
    for supply in nw.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)

        p_max = np.inf
        p_max = min_vals[0]["inlet_p"]
        # for pipe in supply.node.connections["Pipe"]:
        #     p_max = min(pipe.pressure_ASME_MPa, p_max)

        new_supply["pressure_mpa_g"].append(p_max)
        new_supply["flowrate_MW"].append("")

    # Make new demands
    col_names = ["demand_name", "node_name", "flowrate_MW"]
    new_demand = {x: [] for x in col_names}
    for demand in nw.demand_nodes.values():
        new_demand["demand_name"].append(demand.name)
        new_demand["node_name"].append(demand.node.name)
        new_demand["flowrate_MW"].append(demand.flowrate_MW)

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
        p_max = np.inf
        for pipe in supply.node.connections["Pipe"]:
            p_max = min(pipe.pressure_ASME_MPa, p_max)

        new_comps["pressure_out_mpa_g"].append(p_max)

    if add_supply_comp_min:
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
        new_comps["pressure_out_mpa_g"].insert(0, p_max)

        # Add new node
        new_nodes["node_name"].insert(0, "Supply compressor from_node")
        new_nodes["p_max_mpa_g"].insert(0, p_max)

        # Update supply node
        new_supply["node_name"][-1] = "Supply compressor from_node"
        new_supply["pressure_mpa_g"][-1] = sn.pressure_mpa

    # Make new composition
    col_names = ["SPECIES", "X"]
    new_composition = {x: [] for x in col_names}
    for species, x in nw.composition.x.items():
        new_composition["SPECIES"].append(species)
        new_composition["X"].append(x)

    # Format pipes for result file
    new_pipes_f = {}
    for p_i in min_vals:
        new_pipes_f[f"pipe_segment_{p_i}"] = min_vals[p_i]
        new_pipes_f[f"pipe_segment_{p_i}"]["lengths"] = min_vals[p_i]["loop_length"]
        new_pipes_f[f"pipe_segment_{p_i}"]["costs"] = min_vals[p_i]["mat_costs"]
        new_pipes_f[f"pipe_segment_{p_i}"]["name"] = added_pipe_names[p_i]
        new_pipes_f[f"pipe_segment_{p_i}"]["ps"] = p_i

    new_pipes = DataFrame(new_pipes)
    new_nodes = DataFrame(new_nodes)
    new_comps = DataFrame(new_comps)
    new_supply = DataFrame(new_supply)
    new_demand = DataFrame(new_demand)
    new_composition = DataFrame(new_composition)

    # Remake file
    with ExcelWriter(new_filename) as writer:
        new_pipes.to_excel(writer, sheet_name="PIPES", index=False)
        new_nodes.to_excel(writer, sheet_name="NODES", index=False)
        new_comps.to_excel(writer, sheet_name="COMPRESSORS", index=False)
        new_supply.to_excel(writer, sheet_name="SUPPLY", index=False)
        new_demand.to_excel(writer, sheet_name="DEMAND", index=False)
        new_composition.to_excel(writer, sheet_name="COMPOSITION", index=False)

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
    for i in min_vals.values():
        # Ignore 0 loop length loops
        if i["loop_length"] == 0:
            continue
        # Combine by DN,sch,grade
        combined_ind = f"{i['DN']};;{i['schedules']};;{i['grades']}"
        # if combined_ind in combined_pipe["D_S_G"]:
        #     dsg_ind = combined_pipe["D_S_G"].index(combined_ind)
        #     combined_pipe["length"][dsg_ind] += i["loop_length"]
        #     combined_pipe["mat_cost"][dsg_ind] += i["mat_costs"]
        # else:
        combined_pipe["DN"].append(i["DN"])
        combined_pipe["sch"].append(i["schedules"])
        combined_pipe["grade"].append(i["grades"])
        combined_pipe["D_S_G"].append(combined_ind)
        combined_pipe["length"].append(i["loop_length"])
        combined_pipe["mat_cost"].append(i["mat_costs"])

    return new_pipes_f, combined_pipe


def make_loop_network(
    l_loop: float,
    composition: bp_plc.Composition,
    p_in: float,
    offtakes: list,
    all_mdot: list,
    d_main: float,
    d_loop: float,
    roughness_mm: float,
    seg_compressor: list,
    prev_ASME_pressure: float,
    comps_elec: bool,
    eos: bp_plc.eos._EOS_OPTIONS = "rk",
    thermo_curvefit: bool = False,
) -> tuple:
    """
    Create new network to simulate the parallel looped segment
    """
    # Make inlet node
    n_ds_in = bp_plc.Node(name="in", X=composition)
    nodes = {n_ds_in.name: n_ds_in}
    # Initialize pipes, demands, supplies
    pipes = {}
    demands = {}
    supplys = {"supply": bp_plc.Supply_node(node=n_ds_in, pressure_mpa=p_in)}
    compressors = {}

    HHV = composition.HHV

    # Loop through offtakes
    cumsum_offtakes = np.insert(np.cumsum(offtakes), 0, 0)
    l_done = False
    prev_node = n_ds_in
    node_index = 1
    for ot_i, ot_mdot in enumerate(all_mdot):
        name = f"ot_{ot_i}"
        nodes[name] = bp_plc.Node(name=name, X=composition)
        node_index += 1
        d_name = f"demand_{ot_i}"
        demands[d_name] = bp_plc.Demand_node(
            node=nodes[name],
            flowrate_MW=ot_mdot * HHV,
        )
        # If the loop is within the bounds
        if not l_done and abs(cumsum_offtakes[ot_i + 1] - l_loop) < 0.01:
            # Make loop cxn node
            l_cxn_name = name

            p_name = "loop_2_l_cxn"
            pipes[p_name] = bp_plc.Pipe(
                name=p_name,
                from_node=n_ds_in,
                to_node=nodes[l_cxn_name],
                diameter_mm=d_loop,
                length_km=l_loop,
                roughness_mm=roughness_mm,
            )

            l_done = True
        if not l_done and cumsum_offtakes[ot_i] < l_loop < cumsum_offtakes[ot_i + 1]:
            # Make loop cxn node
            l_cxn_name = "loop_cxn"
            nodes[l_cxn_name] = bp_plc.Node(name=l_cxn_name, X=composition)
            node_index += 1
            p_name = "main_2_l_cxn"
            pipes[p_name] = bp_plc.Pipe(
                name=p_name,
                from_node=prev_node,
                to_node=nodes[l_cxn_name],
                diameter_mm=d_main,
                length_km=l_loop - cumsum_offtakes[ot_i],
                roughness_mm=roughness_mm,
            )
            p_name = "loop_2_l_cxn"
            pipes[p_name] = bp_plc.Pipe(
                name=p_name,
                from_node=n_ds_in,
                to_node=nodes[l_cxn_name],
                diameter_mm=d_loop,
                length_km=l_loop,
                roughness_mm=roughness_mm,
            )
            p_name = "l_cxn_2_main"
            pipes[p_name] = bp_plc.Pipe(
                name=p_name,
                from_node=nodes[l_cxn_name],
                to_node=nodes[name],
                diameter_mm=d_main,
                length_km=cumsum_offtakes[ot_i + 1] - l_loop,
                roughness_mm=roughness_mm,
            )
            prev_node = nodes[name]

            l_done = True
        else:
            p_name = f"main_{ot_i}"
            pipes[p_name] = bp_plc.Pipe(
                name=p_name,
                from_node=prev_node,
                to_node=nodes[name],
                diameter_mm=d_main,
                length_km=offtakes[ot_i],
                roughness_mm=roughness_mm,
            )
            prev_node = nodes[name]

    end_node = nodes[name]

    # If a compressor exists in the segment
    if seg_compressor:
        comp_orig = seg_compressor[0]
        comp_name = "segment_compressor"

        # Keep the final node as is for pipe connections as is, but then add another node after.
        # Compressor will compress to this node. Demand node will be updated

        # Add node after compressor
        final_node_name = "final_node"
        nodes[final_node_name] = bp_plc.Node(name=final_node_name, X=composition)
        # Update demand to be the final node

        final_demand_node_name = f"demand_{len(all_mdot)-1}"
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

    looping = BlendPATH_network(
        name="looping",
        pipes=pipes,
        nodes=nodes,
        demand_nodes=demands,
        supply_nodes=supplys,
        compressors=compressors,
        composition=composition,
    )
    looping.set_eos(eos=eos)
    looping.set_thermo_curvefit(thermo_curvefit)
    return looping, end_node, n_ds_in.connections["Pipe"], compressors


def get_loop_length(
    composition: bp_plc.Composition,
    d_main: float,
    d_loop: float,
    l_total: float,
    p_in: float,
    p_out_target: float,
    offtakes: list,
    offtakes_mdot: list,
    roughness_mm: float,
    seg_compressor: list,
    costing_params: bp_cost.Costing_params,
    design_params: Design_params,
    prev_ASME_pressure: float,
    eos: bp_plc.eos._EOS_OPTIONS = "rk",
    thermo_curvefit: bool = False,
) -> tuple:
    """
    Determine the loop length to satisfy constraints
    """
    l_bounds = [0, l_total]

    l_vals = []
    p_vals = []

    all_mdot = offtakes_mdot

    comps_elec = design_params.existing_comp_elec

    # Test boundary conditions first
    for l_loop in l_bounds:
        looping, end_node, end_pipes, compressors = make_loop_network(
            l_loop=l_loop,
            composition=composition,
            p_in=p_in,
            offtakes=offtakes,
            all_mdot=all_mdot,
            d_main=d_main,
            d_loop=d_loop,
            roughness_mm=roughness_mm,
            eos=eos,
            thermo_curvefit=thermo_curvefit,
            seg_compressor=seg_compressor,
            comps_elec=comps_elec,
            prev_ASME_pressure=prev_ASME_pressure,
        )
        try:
            looping.solve()
        except ValueError as err_val:
            p_vals.append(0)
            l_vals.append(l_loop)
            if err_val.args[0] in ["Negative pressure", "Pressure below threshold"]:
                # If negative pressure is achieved with 100% looping then
                # less looping will only make it more negative (higher pressure drop)
                # Thus this diameter/grade is not valid
                if l_loop == l_total:
                    return returning_loop_length(
                        np.nan, end_pipes, 0, costing_params, design_params
                    )
            continue
        p_vals.append(end_node.pressure)
        l_vals.append(l_loop)
        # If 0 looping, and outlet pressure is greater than target, then no looping is the cheapest option
        if end_node.pressure > p_out_target and l_loop == 0:
            return returning_loop_length(
                0, end_pipes, compressors, costing_params, design_params
            )
        # If end pressure is less than target at 100% looping, then this combo is not a solution, since even with
        # 100% looping, there is too much pressure drop
        if end_node.pressure < p_out_target and l_loop == l_total:
            return returning_loop_length(
                np.nan, end_pipes, 0, costing_params, design_params
            )

    iter = 0
    err = np.inf
    while err > gl.SOLVER_TOL:
        if len(p_vals) > 1:
            if l_vals[-1] - l_vals[-2] == np.inf or p_vals[-1] == 0:
                l_loop = np.mean(l_vals[-2:])
                l_vals.pop()
                p_vals.pop()
            else:
                slope = (p_vals[-1] - p_vals[-2]) / (l_vals[-1] - l_vals[-2])
                intercept = p_vals[-1] - slope * l_vals[-1]
                l_loop = (p_out_target - intercept) / slope
                if l_loop <= 0:
                    slope = (p_vals[-1] - p_vals[0]) / (l_vals[-1] - l_vals[0])
                    intercept = p_vals[-1] - slope * l_vals[-1]
                    l_loop = (p_out_target - intercept) / slope
        else:
            l_loop = np.mean(l_bounds)

        # If it is close to the full loop just return full loop
        if abs(l_loop - l_total) < gl.PL_LEN_TOL:
            return returning_loop_length(
                l_total, end_pipes, compressors, costing_params, design_params
            )
        # If it is close to zero loop and 0 wasn't a solution, then not a solution
        if abs(l_loop) < gl.PL_LEN_TOL:
            return returning_loop_length(
                np.nan, end_pipes, 0, costing_params, design_params
            )

        # If the bounds are too close then break out
        if len(l_vals) > 1 and abs(l_vals[-1] - l_vals[-2]) < gl.PL_LEN_TOL:
            return returning_loop_length(
                l_loop, end_pipes, compressors, costing_params, design_params
            )

        looping, end_node, end_pipes, compressors = make_loop_network(
            l_loop=l_loop,
            composition=composition,
            p_in=p_in,
            offtakes=offtakes,
            all_mdot=all_mdot,
            d_main=d_main,
            d_loop=d_loop,
            roughness_mm=roughness_mm,
            eos=eos,
            thermo_curvefit=thermo_curvefit,
            seg_compressor=seg_compressor,
            comps_elec=comps_elec,
            prev_ASME_pressure=prev_ASME_pressure,
        )

        # If negative pressure, solve will provide a Value error
        # If so, then the loop needs to be longer
        # So the lower bound is increased to the current loop length
        try:
            looping.solve()
        except ValueError as err_val:
            if err_val.args[0] in [
                "Negative pressure",
                "Pressure below threshold",
                f"Could not converge in {gl.MAX_ITER} iterations",
            ]:
                l_vals.append(l_loop)
                p_vals.append(0)
            else:
                pass
        else:
            outlet_p = end_node.pressure

            err = abs((p_out_target - outlet_p) / p_out_target)  # In Pa

            l_vals.append(l_loop)
            p_vals.append(end_node.pressure)

        if iter > gl.MAX_ITER:
            raise ValueError(f"Solver could not solve in {gl.MAX_ITER} iterations")
        iter += 1

    return returning_loop_length(
        l_loop, end_pipes, compressors, costing_params, design_params
    )


def returning_loop_length(
    length: float,
    end_pipes: dict,
    compressors: dict,
    costing_params: bp_cost.Costing_params,
    design_params: Design_params,
) -> tuple:
    """
    Return the length and mdot of the segment
    """
    # Added this function to sum up the m_dot in to the segment in one place
    mdot = sum([pipe.m_dot for pipe in end_pipes])

    comp_out = {"fuel": 0, "cost": 0, "revamp": 0, "elec": 0}
    if compressors:
        comp = compressors["segment_compressor"]
        fuel_use = comp.fuel_mdot
        comp_cost = comp.get_cap_cost(
            cp=costing_params, to_electric=design_params.existing_comp_elec
        )

        revamped_comp_capex = comp.get_cap_cost(
            cp=costing_params,
            revamp=True,
            to_electric=design_params.existing_comp_elec,
        )
        fuel_use_elec = comp.fuel_electric_W
        comp_out = {
            "fuel": fuel_use,
            "cost": comp_cost,
            "revamp": revamped_comp_capex,
            "elec": fuel_use_elec,
        }

    return length, mdot, comp_out
