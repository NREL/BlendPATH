import csv
import json
from dataclasses import dataclass
from os.path import exists

import numpy as np
import pandas as pd
from importlib_resources import files
from ProFAST import ProFAST

import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components as bp_plc
import BlendPATH.util.pipe_assessment as bp_pa

ANL_COEFS = {
    "GP": {
        "Labor": [10406, 0.20953, -0.08419],
        "Misc": [4944, 0.17351, -0.07621],
        "ROW": [2751, -0.28294, 0.00731],
        "Material": [5813, 0.31599, -0.00376],
    },
    "NE": {
        "Labor": [249131, -0.33162, -0.17892],
        "Misc": [65990, -0.29673, -0.06856],
        "ROW": [83124, -0.66357, -0.07544],
        "Material": [10409, 0.296847, -0.07257],
    },
    "MA": {
        "Labor": [43692, 0.05683, -0.10108],
        "Misc": [14616, 0.16354, -0.16186],
        "ROW": [1942, 0.17394, -0.01555],
        "Material": [9113, 0.279875, -0.00840],
    },
    "GL": {
        "Labor": [58154, -0.14821, -0.10596],
        "Misc": [41238, -0.34751, -0.11104],
        "ROW": [14259, -0.65318, 0.06865],
        "Material": [8971, 0.255012, -0.03138],
    },
    "RM": {
        "Labor": [10406, 0.20953, -0.08419],
        "Misc": [4944, 0.17351, -0.07621],
        "ROW": [2751, -0.28294, 0.00731],
        "Material": [5813, 0.31599, -0.00376],
    },
    "SE": {
        "Labor": [32094, 0.06110, -0.14828],
        "Misc": [11270, 0.19077, -0.13669],
        "ROW": [9531, -0.37284, 0.02616],
        "Material": [6207, 0.38224, -0.05211],
    },
    "PN": {
        "Labor": [32094, 0.06110, -0.14828],
        "Misc": [11270, 0.19077, -0.13669],
        "ROW": [9531, -0.37284, 0.02616],
        "Material": [6207, 0.38224, -0.05211],
    },
    "SW": {
        "Labor": [95295, -0.53848, 0.03070],
        "Misc": [19211, -0.14178, -0.04697],
        "ROW": [72634, -1.07566, 0.05284],
        "Material": [5605, 0.41642, -0.06441],
    },
    "CA": {
        "Labor": [95295, -0.53848, 0.03070],
        "Misc": [19211, -0.14178, -0.04697],
        "ROW": [72634, -1.07566, 0.05284],
        "Material": [5605, 0.41642, -0.06441],
    },
}

COMPR_COSTS = {
    "Material": [3175286, 532.7853, 0.0010416],
    "Labor": [1581740, 299.2887, 0.001142],
    "Misc": [1696686, 184.1443, 0.0018417],
    "Land": [66216.72, 0, 0.0001799],
}


@dataclass
class Costing_params:
    """
    Grouping of cost parameters from BlendPATH_scenario
    """

    h2_price: float
    ng_price: float
    elec_price: float
    region: str
    cf_price: float
    casestudy_name: str
    ili_interval: float
    original_pipeline_cost: float
    pipe_markup: float
    compressor_markup: float
    financial_overrides: float


def get_steel_cost_file(steel_cost_file: str) -> dict:
    """
    Retrieve steel cost file
    """
    with open(steel_cost_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return {row["Steel grade"]: float(row["Price [$/kg]"]) for row in reader}


def get_steel_cost(cp: Costing_params, grade: str, mass: float) -> float:
    """
    Retrive steel cost based on grade
    """
    if grade not in cp.steel_cost.keys():
        raise ValueError(f"{grade} is not a valid steel grade")
    unit_price = cp.steel_cost[grade]
    price = unit_price * mass

    return price


def get_ANL_costs_in_mi(
    diameter_mm: float, length_km: float, region: str, c_type=list
) -> float:
    """
    Get cost correlations from Brown, Reddi, Elgowainy, Int J. Hydrogen Energy, 2022
    """
    d_in = diameter_mm * gl.MM2IN
    l_mi = length_km * gl.KM2MI
    if region not in ANL_COEFS.keys():
        raise ValueError(f"{region} is not a valid region")

    anl_reg = ANL_COEFS[region]

    c = anl_reg[c_type]
    cost_res = c[0] * d_in ** c[1] * l_mi ** c[2]
    return cost_res


def get_pipe_cost_file(pipe_file: str) -> dict:
    """
    Get override pipe cost file
    """
    with open(pipe_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return {
            row["Parameter"]: float(row["Price [$/in/mi]"])
            for row in reader
            if row["Price [$/in/mi]"] != ""
        }


def get_compressor_cost_file(compressor_file: str) -> dict:
    """
    Get override compressor cost file
    """
    with open(compressor_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return {
            row["Parameter"]: float(row["Price [$/hp]"])
            for row in reader
            if row["Price [$/hp]"] != ""
        }


def get_pipe_other_cost(
    cp: Costing_params, d_mm: float, l_km: float, anl_types: list
) -> dict:
    """
    Get material and other costs for pipes
    """
    all_other_cost = {x: 0 for x in ["Mat"] + anl_types}
    if l_km == 0:
        return all_other_cost
    d_in = d_mm * gl.MM2IN
    l_mi = l_km * gl.KM2MI
    per_in_mi_costs = {x: 0 for x in anl_types}
    for x in per_in_mi_costs.keys():
        if x in cp.pipe_cost_override.keys():
            per_in_mi_costs[x] = cp.pipe_cost_override[x]
        else:
            per_in_mi_costs[x] = get_ANL_costs_in_mi(
                diameter_mm=d_mm, length_km=l_km, region=cp.region, c_type=x
            )
        all_other_cost[x] = per_in_mi_costs[x] * d_in * l_mi * cp.pipe_markup

    return all_other_cost


def get_pipe_material_cost(
    cp: Costing_params, di_mm: float, do_mm: float, l_km: float, grade: str
) -> float:
    """
    Pipe material cost
    """
    pipe_vol_m3 = bp_pa.get_pipe_volume(
        diam_i_m=di_mm * gl.MM2M,
        diam_o_m=do_mm * gl.MM2M,
        length_m=l_km * gl.KM2M,
    )
    pipe_mass_kg = bp_pa.get_pipe_mass(volume_m3=pipe_vol_m3)
    pipe_mat_cost = get_steel_cost(cp=cp, grade=grade, mass=pipe_mass_kg)

    return pipe_mat_cost * cp.pipe_markup


def calc_lcot(
    json_file: str,
    capacity: float,
    new_pipe_cap: float,
    comp_cost: float,
    revamped_comp_capex: float,
    supply_comp_capex: float,
    compressor_fuel: float,
    compressor_fuel_elec: float,
    supply_comp_fuel: dict,
    cs_cost: float,
    elec_cost: float,
    meter_cost: float = 0,
    ili_costs: float = 0,
    valve_cost: float = 0,
    original_network_residual_value: float = 0,
    financial_overrides: dict = None,
) -> pd.DataFrame:
    """
    Calaculate the levelized cost of transport with ProFAST
    """
    jsonfilename = f"{json_file}/financial_params.json"
    if not exists(jsonfilename):
        jsonfilename = files("BlendPATH.costing").joinpath(
            "default_financial_params.json"
        )

    pf = ProFAST(jsonfilename)
    if financial_overrides is not None:
        for i, v in financial_overrides.items():
            pf.set_params(i, v)

    gen_inflation = pf.vals["general inflation rate"]

    pf.set_params("capacity", capacity)  # MMBTU/day
    pf.set_params(
        "commodity",
        {
            "name": "Natural Gas-H2 Blend",
            "unit": "MMBTU",
            "initial price": 0.25,
            "escalation": gen_inflation,
        },
    )

    col_names = {
        "new_pipe": "New pipe CapEx",
        "new_comp": "New compression capacity CapEx",
        "refurb_comp": "Refurbished compressor capacity CapEx",
        "supply_comp": "Supply compressor CapEx",
        "meter": "Meter & regulator station modification CapEx",
        "valve": "Valve replacement CapEx",
        "orig_network": "Original network residual value",
        "ili": "In-line inspection",
        "fuel": "Compressor fuel",
        "fuel_elec": "Compressor fuel (electric)",
        "supply_fuel_elec": "Supply commpressor fuel (electric)",
        "supply_fuel_gas": "Supply commpressor fuel",
    }

    # Add compressor fuel (gas and electric)
    pf.add_feedstock(
        name=col_names["fuel"],
        usage=compressor_fuel,
        unit="MMBTU",
        cost=cs_cost,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name=col_names["fuel_elec"],
        usage=compressor_fuel_elec,
        unit="kWh",
        cost=elec_cost,
        escalation=gen_inflation,
    )

    pf.add_feedstock(
        name=col_names["supply_fuel_elec"],
        usage=supply_comp_fuel["elec"],
        unit="kWh",
        cost=elec_cost,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name=col_names["supply_fuel_gas"],
        usage=supply_comp_fuel["gas"],
        unit="MMBTU",
        cost=cs_cost,
        escalation=gen_inflation,
    )

    depr_period = pf.vals["operating life"]
    # Add new pipe capex
    pf.add_capital_item(
        name=col_names["new_pipe"],
        cost=new_pipe_cap,
        depr_type="Straight line",
        depr_period=depr_period,
        refurb=[0],
    )

    # Add new compressors
    pf.add_capital_item(
        name=col_names["new_comp"],
        cost=sum(comp_cost),
        depr_type="Straight line",
        depr_period=depr_period,
        refurb=[0],
    )
    # Add revamped compressors
    pf.add_capital_item(
        name=col_names["refurb_comp"],
        cost=sum(revamped_comp_capex),
        depr_type="Straight line",
        depr_period=depr_period,
        refurb=[0],
    )
    # Add supply compressors
    pf.add_capital_item(
        name=col_names["supply_comp"],
        cost=supply_comp_capex,
        depr_type="Straight line",
        depr_period=depr_period,
        refurb=[0],
    )
    # Add meter & regulator station
    pf.add_capital_item(
        name=col_names["meter"],
        cost=meter_cost,
        depr_type="Straight line",
        depr_period=depr_period,
        refurb=[0],
    )
    # Add valve replacement
    pf.add_capital_item(
        name=col_names["valve"],
        cost=valve_cost,
        depr_type="Straight line",
        depr_period=depr_period,
        refurb=[0],
    )
    # Add residual value of old network
    pf.add_capital_item(
        name=col_names["orig_network"],
        cost=original_network_residual_value,
        depr_type="Straight line",
        depr_period=depr_period,
        refurb=[0],
    )

    #   Add inline-inspection costs
    pf.add_fixed_cost(
        name=col_names["ili"],
        usage=1,
        unit="$",
        cost=ili_costs,
        escalation=gen_inflation,
    )

    sol = pf.solve_price()

    price_breakdown = pf.get_cost_breakdown()
    add_lcot_str = [f"LCOT: {x}" for x in col_names.values()]
    prices_all = {
        x: 0
        for x in ["LCOT: Levelized cost of transport"]
        + add_lcot_str
        + ["LCOT: Fixed O&M", "LCOT: Taxes", "LCOT: Financial"]
    }

    prices_all["LCOT: Levelized cost of transport"] = sol["lco"]
    prices_all[f"LCOT: {col_names['fuel']}"] = price_breakdown_cols(
        price_breakdown, [col_names["fuel"]], []
    )
    prices_all[f"LCOT: {col_names['fuel_elec']}"] = price_breakdown_cols(
        price_breakdown, [col_names["fuel_elec"]], []
    )
    prices_all[f"LCOT: {col_names['supply_fuel_gas']}"] = price_breakdown_cols(
        price_breakdown, [col_names["supply_fuel_gas"]], []
    )
    prices_all[f"LCOT: {col_names['supply_fuel_elec']}"] = price_breakdown_cols(
        price_breakdown, [col_names["supply_fuel_elec"]], []
    )

    # Get financial expense associated with equipment
    pos = ["Repayment of debt", "Interest expense", "Dividends paid"]
    neg = ["Inflow of debt", "Inflow of equity"]
    cap_expense = price_breakdown_cols(price_breakdown, pos, neg)

    # Get remaining financial expenses
    pos = ["Non-depreciable assets", "Cash on hand reserve"]
    neg = ["Sale of non-depreciable assets", "Cash on hand recovery"]
    prices_all["LCOT: Financial"] = price_breakdown_cols(price_breakdown, pos, neg)

    # Get tax related
    pos = ["Income taxes payable", "Capital gains taxes payable"]
    neg = ["Monetized tax losses"]
    prices_all["LCOT: Taxes"] = price_breakdown_cols(price_breakdown, pos, neg)

    # Get fixed O&M
    pos = ["Administrative expenses", "Property insurance"]
    neg = []
    prices_all["LCOT: Fixed O&M"] = price_breakdown_cols(price_breakdown, pos, neg)

    # Get inline inspection
    pos = [col_names["ili"]]
    neg = []
    prices_all["LCOT: In-line inspection"] = price_breakdown_cols(
        price_breakdown, pos, neg
    )

    total_capex = {
        col_names["new_pipe"]: new_pipe_cap,
        col_names["orig_network"]: original_network_residual_value,
        col_names["new_comp"]: sum(comp_cost),
        col_names["refurb_comp"]: sum(revamped_comp_capex),
        col_names["supply_comp"]: supply_comp_capex,
        col_names["meter"]: meter_cost,
        col_names["valve"]: valve_cost,
    }

    capex_sum = sum(total_capex.values())
    capex_fraction = {x: 0 for x in total_capex.keys()}
    if capex_sum > 0:
        capex_fraction = {x: y / capex_sum for x, y in total_capex.items()}

    # Get CAPEX and distribute related financial expenses
    for i in total_capex.keys():
        prices_all[f"LCOT: {i}"] = (
            price_breakdown_cols(price_breakdown, [i]) + cap_expense * capex_fraction[i]
        )

    return prices_all


def get_cs_fuel_cost(
    blend: float,
    ng_cost: float,
    h2_cost: float,
    ng_comp: dict,
    json_file: str,
    fin_overrides: dict,
) -> float:
    """
    Calculate the compressor fuel cost in $/MMBTU
    """

    # Get pure H2 HHV,GCV
    pure_h2 = bp_plc.Composition({"H2": 1})
    H2_energy_HHV = pure_h2.HHV  # MJ/kg
    GCV_H2_MJpsm3 = pure_h2.get_GCV()

    # Get pure CH4 GCV
    pure_ch4 = bp_plc.Composition(ng_comp)
    GCV_NG_MJpsm3 = pure_ch4.get_GCV()

    blend_ratio_energy = (blend * GCV_H2_MJpsm3) / (
        (blend * GCV_H2_MJpsm3) + (1 - blend) * GCV_NG_MJpsm3
    )
    # h2_cost_MMBTU = h2_cost / H2_energy_HHV / gl.MJ2MMBTU

    jsonfilename = f"{json_file}/financial_params.json"
    if not exists(jsonfilename):
        jsonfilename = files("BlendPATH.costing").joinpath(
            "default_financial_params.json"
        )
    with open(jsonfilename) as f:
        d = json.load(f)
    var_list = [
        "general inflation rate",
        "analysis start year",
        "installation months",
        "operating life",
    ]
    fin_params = {
        x: d["variables"][x] if x not in fin_overrides.keys() else fin_overrides[x]
        for x in var_list
    }
    make_range = range(
        fin_params["analysis start year"],
        fin_params["analysis start year"]
        + fin_params["operating life"]
        + int(fin_params["installation months"] / 12)
        + 10,
    )
    infl = fin_params["general inflation rate"]

    if isinstance(ng_cost, dict):
        ng_years = set(ng_cost.keys())
    else:
        ng_years = set(make_range)
        ng_cost = {year: ng_cost * (1 + infl) ** i for i, year in enumerate(make_range)}
    if isinstance(h2_cost, dict):
        h2_cost_MMBTU = {
            year: val / H2_energy_HHV / gl.MJ2MMBTU for year, val in h2_cost.items()
        }
        h2_years = set(h2_cost_MMBTU.keys())
    else:
        h2_cost_MMBTU = h2_cost / H2_energy_HHV / gl.MJ2MMBTU
        h2_years = set(make_range)
        h2_cost_MMBTU = {
            year: h2_cost_MMBTU * (1 + infl) ** i for i, year in enumerate(make_range)
        }

    key_intersect = ng_years.union(h2_years)

    ng_price_all = {
        year: (ng_cost[year] if year in ng_cost.keys() else 0) for year in key_intersect
    }
    h2_price_all = {
        year: (h2_cost_MMBTU[year] if year in h2_cost_MMBTU.keys() else 0)
        for year in key_intersect
    }

    def cs_func(ng, h2, b):
        return ng * (1 - b) + h2 * b

    cs_cost = {
        key: cs_func(ng_price_all[key], h2_price_all[key], blend_ratio_energy)
        for key in key_intersect
    }

    return cs_cost


def get_compressor_cost(
    cp: Costing_params,
    avg_station_cap: float,
    num_comp_stations: float,
    revamped_ratio: float = 1,
) -> float:
    """
    Compressor cost correlation
    """
    cols = ["Material", "Labor", "Misc", "Land"]
    rui_amsymptote = {"Material": 694.09, "Labor": 400.24, "Misc": 306.65, "Land": 7.88}
    comp_costs = {x: 0 for x in cols}
    for cost_type in cols:
        if cost_type in cp.comp_cost_override.keys():
            comp_costs[cost_type] = avg_station_cap * cp.comp_cost_override[cost_type]
        else:
            # Rui ampytotes to 1400 $/hp above 30,000 hp
            if avg_station_cap > 30_000:
                comp_costs[cost_type] = rui_amsymptote[cost_type] * avg_station_cap
            else:
                station_cap_array = [avg_station_cap**x for x in [0, 1, 2]]

                comp_costs[cost_type] = np.dot(
                    COMPR_COSTS[cost_type], station_cap_array
                )

            CPI_ratio = 596.2 / 575.4  # Go from 2008 to 2020
            comp_costs[cost_type] *= CPI_ratio

    if revamped_ratio != 1:
        comp_costs[cols.index("Land")] = 0

    comp_cost_i = sum(comp_costs.values()) * revamped_ratio * num_comp_stations
    return comp_cost_i * cp.compressor_markup


def price_breakdown_cols(df: pd.DataFrame, pos: list = None, neg: list = None) -> float:
    """
    Sum up positive and negative colums for cost breakdown
    """
    if pos is None:
        pos = []
    if neg is None:
        neg = []
    return (
        df.loc[df["Name"].isin(pos), "NPV"].sum()
        - df.loc[df["Name"].isin(neg), "NPV"].sum()
    )


def meter_reg_station_cost(cp: Costing_params, demands_MW: list) -> float:
    """
    Returns meter, regulator, and GC modification costs
    """

    demand_mmbtu = np.array(demands_MW) * gl.MW2MMBTUDAY

    cost_meters = meter_replacement_cost(cp=cp, demand_mmbtu=demand_mmbtu)
    cost_gc = GC_cost(cp=cp, demand_mmbtu=demand_mmbtu)
    cost_reg = regulator_cost(cp=cp, demand_mmbtu=demand_mmbtu)

    return cost_meters + cost_gc + cost_reg


def meter_replacement_cost_file(meter_cost_file: str) -> tuple:
    """
    Retrieve default meter cost file
    """
    with open(meter_cost_file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        for row in reader:
            m = float(row["m [2020$/MMBTU-day]"])
            b = float(row["b [2020$]"])
            return (m, b)


def meter_replacement_cost(cp: Costing_params, demand_mmbtu: list) -> float:
    """
    Get meter replacement costs
    """
    m, b = cp.meter_cost

    return np.sum(m * demand_mmbtu + b)


def GC_cost_file(gc_cost_file) -> float:
    """
    Read in GC cost file
    """
    with open(gc_cost_file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        for row in reader:
            installed_cost = float(row["Installed cost [2020$]"])
            return installed_cost


def GC_cost(cp: Costing_params, demand_mmbtu: list) -> float:
    """
    Get gas chromatography modifications/addition cost

    """
    installed_cost = cp.gc_cost  # [2020$]

    return installed_cost * len(demand_mmbtu)


def regulator_cost_file(regulator_cost_file: str) -> tuple:
    """
    Retrieve default regulator cost file
    """
    with open(regulator_cost_file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        for row in reader:
            capacity_ref = float(row["Capacity [MMBTU/day]"])
            installed_cost_ref = float(row["Installed regulator cost [2020$]"])
            return (capacity_ref, installed_cost_ref)


def regulator_cost(cp: Costing_params, demand_mmbtu: list) -> float:
    """
    Installed pressure regulator costs

    """
    capacity_ref = 311400  # [MMBTU/day]
    installed_cost_ref = 2248722  # [2020$]
    capacity_ref, installed_cost_ref = cp.regulator_cost

    regulator_runs = np.ceil(demand_mmbtu / capacity_ref)
    replacement_cost = regulator_runs * installed_cost_ref

    return sum(replacement_cost)


def ili_cost(cp: Costing_params, pipe_added: list) -> float:
    """
    ILI costs from tabulated values
    """

    ili_cost = sum([cp.ili_cost[dn] * length for dn, length in pipe_added])

    return ili_cost / cp.ili_interval


def ili_costs_file(valve_cost_file: str) -> dict:
    """
    Import ILI costs file
    """

    with open(valve_cost_file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        ili_dict = {int(x["DN"]): float(x["ILI cost [2020$/mi]"]) for x in reader}
    return ili_dict


def valve_replacement_cost_file(file: str) -> dict:
    """
    Get valve replacement costs:
    """
    with open(file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        buried = {}
        above = {}
        for row in reader:
            if row["Install type"] == "Buried":
                buried[int(row["DN"])] = int(row["Installed valve cost [2020$]"])
            elif row["Install type"] == "Above ground, bolted":
                above[int(row["DN"])] = int(row["Installed valve cost [2020$]"])
    return {"buried": buried, "above": above}


def valve_replacement_cost(
    cp: Costing_params, pipe_added: list, loc_class: int, v_type: str = "buried"
) -> float:
    """
    Calculate replacement cost of valves
    """
    #   Establish valve spacing interval as function of Class Location as per ASME B31.12
    # Dict format with key being class location
    ASMEB3112_valve_spacing = {
        1: 20,
        2: 15,
        3: 10,
        4: 5,
    }

    if loc_class not in ASMEB3112_valve_spacing.keys():
        raise ValueError(
            f"Location class value is not valid. {loc_class} was given. Must be in {ASMEB3112_valve_spacing.keys()}"
        )
    spacing = ASMEB3112_valve_spacing[loc_class]

    if v_type not in ["buried", "above"]:
        raise ValueError(f"Valve type must be 'buried' or 'above', {v_type} was given")
    ref_data = cp.valve_cost[v_type]

    valve_cost = sum(
        [np.floor(length / spacing) * ref_data[dn] for dn, length in pipe_added]
    )

    return valve_cost
