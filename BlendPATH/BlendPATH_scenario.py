import csv
import os
import re
from typing import Literal, get_args

import numpy as np
import pandas as pd
from importlib_resources import files
import cantera as ct

import BlendPATH.costing.costing as bp_cost
import BlendPATH.modifications as bp_mod
import BlendPATH.util.pipe_assessment as bp_pa
import BlendPATH.network.pipeline_components as bp_plc

from . import Global as gl
from .network import BlendPATH_network, Design_params
from .util.disclaimer import disclaimer_message

_MOD_TYPES = Literal[
    "direct_replacement", "parallel_loop", "additional_compressors", "dr", "pl", "ac"
]


class BlendPATH_scenario:
    """
    BlendPATH_scenario:
    ---------------
    Description: This is a BlendPATH scenario that does the modification and analysis on a BlendPATH network

    Required Parameters:
    ---------------
    casestudy_name: str = Path to directory of case study files
    """

    def __init__(self, casestudy_name: str, **kwargs) -> None:

        # Assign class variables
        self.casestudy_name = casestudy_name

        # Check for default inputs file
        temp_inputs = {
            "blend": 0.1,
            "location_class": 1,
            "T_rating": 1,
            "joint_factor": 1,
            "design_option": "b",
            "ng_price": 7.39,
            "h2_price": 4.40756,
            "elec_price": 0.07,
            "region": "GP",
            "design_CR": [1.2, 1.4, 1.6, 1.8, 2.0],
            "final_outlet_pressure_mpa_g": 2,
            "verbose": True,
            "results_dir": "out",
            "eos": "rk",
            "ili_interval": 3,
            "original_pipeline_cost": 0,
            "new_compressors_electric": True,
            "existing_compressors_to_electric": True,
            "new_comp_eta_s": 0.78,
            "new_comp_eta_s_elec": 0.88,
            "new_comp_eta_driver": 0.357,
            "new_comp_eta_driver_elec": np.nan,
            "pipe_markup": 1,
            "compressor_markup": 1,
            "financial_overrides": {},
            "filename_suffix": "",
        }
        self.read_inputs(temp_inputs)

        # Assign variables - check with user input and input file

        # First assign all defaults
        # Then assign case study defaults
        # Then assign locally put in values
        asme_param_names = ["location_class", "T_rating", "joint_factor"]
        costing_param_names = [
            "h2_price",
            "ng_price",
            "elec_price",
            "region",
            "ili_interval",
            "original_pipeline_cost",
            "pipe_markup",
            "compressor_markup",
            "financial_overrides",
        ]
        design_param_names = [
            "final_outlet_pressure_mpa_g",
            "design_CR",
            "new_compressors_electric",
            "existing_compressors_to_electric",
            "new_comp_eta_s",
            "new_comp_eta_s_elec",
            "new_comp_eta_driver",
            "new_comp_eta_driver_elec",
        ]
        vars = temp_inputs.keys()
        for var in vars:
            temp_inputs[var] = kwargs.get(var, temp_inputs[var])
            if (
                var in asme_param_names
                or var in costing_param_names
                or var in design_param_names
            ):
                continue
            setattr(self, var, temp_inputs[var])
        self.costing_params = bp_cost.Costing_params(
            temp_inputs["h2_price"],
            temp_inputs["ng_price"],
            temp_inputs["elec_price"],
            temp_inputs["region"],
            0,
            self.casestudy_name,
            temp_inputs["ili_interval"],
            temp_inputs["original_pipeline_cost"],
            temp_inputs["pipe_markup"],
            temp_inputs["compressor_markup"],
            temp_inputs["financial_overrides"],
        )
        self.design_params = Design_params(
            temp_inputs["final_outlet_pressure_mpa_g"],
            temp_inputs["design_CR"],
            temp_inputs["new_compressors_electric"],
            temp_inputs["existing_compressors_to_electric"],
            temp_inputs["new_comp_eta_s"],
            temp_inputs["new_comp_eta_s_elec"],
            temp_inputs["new_comp_eta_driver"],
            temp_inputs["new_comp_eta_driver_elec"],
        )
        self.ASME_params = bp_pa.ASME_consts(
            int(temp_inputs["location_class"]),
            temp_inputs["T_rating"],
            temp_inputs["joint_factor"],
        )

        # Checkcost overrides
        self.check_overrides()

        # Print disclaimer message
        self.bp_print(disclaimer_message())

        # Initialize BlendPATH_network
        self.network = BlendPATH_network.import_from_file(
            f"{self.casestudy_name}/network_design.xlsx"
        )
        self.network.set_eos(self.eos)

        # Set design option
        self.update_design_option(self.design_option, init=True)

        # Segmentation
        self.network.pipe_segments = self.network.segment_pipe()
        self.network.reassign_offtakes()
        self.network.pipe_segments[-1].mdot_out = self.network.pipe_segments[
            -1
        ].offtake_mdots[-1]

        # Blend in hydrogen
        self.blendH2(self.blend, self.costing_params.h2_price, True)

    def read_inputs(self, temp_inputs: dict) -> None:
        """
        Checks for a local default_inputs.csv and add those to temp inputs
        """
        valid_params = temp_inputs.keys()
        # costing_params = ["ng_price", "h2_price", "region", "ili_interval"]
        default_inputs_filepath = f"{self.casestudy_name}/default_inputs.csv"
        if os.path.exists(default_inputs_filepath):
            with open(default_inputs_filepath, newline="") as csvfile:
                next(csvfile)
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] not in valid_params:
                        raise ValueError(
                            f"{row[0]} is not a valid input file parameter"
                        )
                    val = row[1]
                    # Check if it is a number or a bool
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    if row[0] in [
                        "verbose",
                        "new_compressors_electric",
                        "existing_compressors_to_electric",
                    ]:
                        val = val in ["TRUE", "True", "true", 1]
                    if row[0] == "design_CR":
                        # Allows for list in a single cell, or as a regular csv input
                        val = [
                            float(x)
                            for x in re.sub(r"[\[\]]", "", ",".join(row[1:])).split(",")
                        ]

                    temp_inputs[row[0]] = val

    def check_overrides(self) -> None:
        """
        Check if override costing values are empolyed
        """
        # Check if overrides directory exists
        overrides_dir = f"{self.casestudy_name}/overrides"
        # if not os.path.exists(overrides_dir):
        #     return

        # Check valve overrides
        valve_file = f"{overrides_dir}/valve_costs.csv"
        if os.path.exists(valve_file):
            self.bp_print("Valve cost override found")
        else:
            valve_file = files("BlendPATH.costing").joinpath("valve_costs.csv")
        self.costing_params.valve_cost = bp_cost.valve_replacement_cost_file(valve_file)

        # Check GC cost overrides
        gc_file = f"{overrides_dir}/GC_cost.csv"
        if os.path.exists(gc_file):
            self.bp_print("GC cost override found")
        else:
            gc_file = files("BlendPATH.costing").joinpath("GC_cost.csv")
        self.costing_params.gc_cost = bp_cost.GC_cost_file(gc_file)

        # Check ILI cost overrides
        ili_file = f"{overrides_dir}/inline_inspection_costs.csv"
        if os.path.exists(ili_file):
            self.bp_print("Inline inspection cost override found")
        else:
            ili_file = files("BlendPATH.costing").joinpath(
                "inline_inspection_costs.csv"
            )
        self.costing_params.ili_cost = bp_cost.ili_costs_file(ili_file)

        # Check regulator cost overrides
        regulator_file = f"{overrides_dir}/regulator_costs.csv"
        if os.path.exists(regulator_file):
            self.bp_print("Regulator cost override found")
        else:
            regulator_file = files("BlendPATH.costing").joinpath("regulator_costs.csv")
        self.costing_params.regulator_cost = bp_cost.regulator_cost_file(regulator_file)

        # Check meter cost overrides
        meter_file = f"{overrides_dir}/meter_replacement_cost_regression_parameters.csv"
        if os.path.exists(meter_file):
            self.bp_print("Meter cost override found")
        else:
            meter_file = files("BlendPATH.costing").joinpath(
                "meter_replacement_cost_regression_parameters.csv"
            )
        self.costing_params.meter_cost = bp_cost.meter_replacement_cost_file(meter_file)

        # Check steel cost overrides
        steel_cost_file = f"{overrides_dir}/steel_costs_per_kg.csv"
        if os.path.exists(steel_cost_file):
            self.bp_print("Steel cost override found")
        else:
            steel_cost_file = files("BlendPATH.costing").joinpath(
                "steel_costs_per_kg.csv"
            )
        self.costing_params.steel_cost = bp_cost.get_steel_cost_file(steel_cost_file)

        # Check pipe cost overrides
        pipe_cost_file = f"{overrides_dir}/pipe_cost.csv"
        self.costing_params.pipe_cost_override = {}
        if os.path.exists(pipe_cost_file):
            self.bp_print("Pipe cost override found")
            self.costing_params.pipe_cost_override = bp_cost.get_pipe_cost_file(
                pipe_cost_file
            )

        # Check compressor cost overrides
        compressor_cost_file = f"{overrides_dir}/compressor_cost.csv"
        self.costing_params.comp_cost_override = {}
        if os.path.exists(compressor_cost_file):
            self.bp_print("Compressor cost override found")
            self.costing_params.comp_cost_override = bp_cost.get_compressor_cost_file(
                compressor_cost_file
            )

    def run_mod(self, mod_type: _MOD_TYPES, design_option: str = "b") -> float:
        """
        Run a modification method on the network using the scenario parameters

        Parameters:
        ------------
        mod_type:str = direct_replacement, parallel_loop, additional_compressors, DR, PL, or AC
        design_option:str = 'b' - Design option for new pipe
        """
        self.bp_print(f"Running modification method: {mod_type}")
        mod_type_filter = mod_type.lower().replace(" ", "_")
        if mod_type_filter not in get_args(_MOD_TYPES):
            raise ValueError(
                f"Modification type must be one {get_args(_MOD_TYPES)}, instead {mod_type} was given"
            )
        mod_type_shorten = {
            "direct_replacement": "dr",
            "parallel_loop": "pl",
            "additional_compressors": "ac",
        }
        self.mod_type = (
            mod_type_shorten[mod_type_filter]
            if mod_type_filter in mod_type_shorten
            else mod_type_filter
        )
        # Update design option for new pipe
        self.design_option_new = bp_pa.check_design_option(design_option)

        # Check if results file exists
        full_results_dir = f"{self.casestudy_name}/{self.results_dir}"
        network_dir = f"{full_results_dir}/NetworkFiles"
        for filepaths in [
            f"{full_results_dir}",
            network_dir,
            f"{full_results_dir}/ResultFiles",
        ]:
            if not os.path.exists(filepaths):
                os.makedirs(filepaths)

        new_filename = f"{self.mod_type.upper()}_{self.blend}_{self.design_option}_"
        new_filename_full = (
            f"{network_dir}/{new_filename}network_design{self.filename_suffix}.xlsx"
        )

        # Pass on to modification type selected
        if self.mod_type in ["direct_replacement", "dr"]:
            all_new_pipes, cap_cost = bp_mod.direct_replacement(
                network=self.network,
                ASME_params=self.ASME_params,
                design_option=design_option,
                new_filename=new_filename_full,
                design_params=self.design_params,
                costing_params=self.costing_params,
            )
            anl_types = ["Labor", "Misc"]
            n_comps = [0]
            l_comps = []

        if self.mod_type in ["parallel_loop", "pl"]:
            all_new_pipes, cap_cost = bp_mod.parallel_loop(
                self.network,
                self.ASME_params,
                self.design_params,
                design_option,
                new_filename_full,
                self.costing_params,
            )
            anl_types = ["Labor", "Misc", "ROW"]
            n_comps = [0]
            l_comps = []

        if self.mod_type in ["additional_compressors", "ac"]:
            cap_cost = {"D_S_G": [], "length": [], "total cost": 0}
            all_new_pipes = {}
            n_comps, l_comps = bp_mod.additional_compressors(
                self.network,
                self.design_params,
                new_filename_full,
                self.costing_params,
            )

        # Import modified network and run simulation
        # Subsegments not currently used in blendpath
        new_network = BlendPATH_network.import_from_file(
            new_filename_full, subsegments=False
        )
        new_network.set_eos(self.eos)

        iters = 0
        while iters < 5:
            try:
                new_network.solve(c_relax=gl.RELAX_FACTOR * 1.05**iters)
                break
            except (ValueError, ct.CanteraError):
                iters += 1
                continue
        else:
            raise ValueError("Cound not solve new network")

        # Calculate ANL costs
        # and get new pipe capex cost
        new_pipe_cap = 0
        for i in range(len(cap_cost["D_S_G"])):
            cap_cost["anl_costs"].append(
                bp_cost.get_pipe_other_cost(
                    cp=self.costing_params,
                    d_mm=cap_cost["DN"][i],
                    l_km=cap_cost["length"][i],
                    anl_types=anl_types,
                )
            )
            mat_cost = cap_cost["mat_cost"][i]
            anl_cost = sum([cap_cost["anl_costs"][i][x] for x in anl_types])
            new_pipe_cap += mat_cost + anl_cost
            cap_cost["total cost"] = new_pipe_cap

        # Get fuel usage
        all_fuel_MW = 0
        all_fuel_elec_kW = 0
        supply_comp_fuel = {"gas": 0, "elec": 0}
        for comp in new_network.compressors.values():
            if comp.name == "Supply compressor":
                supply_comp_fuel = {
                    "gas": comp.fuel_w * gl.W2MW,
                    "elec": comp.fuel_electric_W / gl.KW2W,
                }
                continue
            all_fuel_elec_kW += comp.fuel_electric_W / gl.KW2W
            all_fuel_MW += comp.fuel_w * gl.W2MW

        # Get compressor capex of added compressors
        # Get revamped compressor cost
        comp_capex = []
        revamped_comp_capex = []
        supply_comp_capex = 0

        for comp in new_network.compressors.values():
            # Separate out supply compressor if it exists
            if comp.name == "Supply compressor":
                supply_comp_capex += comp.get_cap_cost(
                    cp=self.costing_params,
                    to_electric=self.design_params.existing_comp_elec,
                )
                comp.revamp_cost = 0
                continue
            c_capex = comp.get_cap_cost(
                cp=self.costing_params,
                to_electric=self.design_params.existing_comp_elec,
            )
            c_capex_revamp = comp.get_cap_cost(
                cp=self.costing_params,
                revamp=True,
                to_electric=self.design_params.existing_comp_elec,
            )
            comp_capex.append(c_capex)
            revamped_comp_capex.append(c_capex_revamp)

        # Get compressor fuel cost
        cf_price = bp_cost.get_cs_fuel_cost(
            self.blend,
            self.costing_params.ng_price,
            self.costing_params.h2_price,
            self.network.composition.pure_x,
            self.casestudy_name,
            self.costing_params.financial_overrides,
        )

        # Organize all pipe for ILI and valve costs. Aggregate on DN
        all_pipes_len = []
        pipe_dns_lens = {}
        for pipe in new_network.pipes.values():
            if pipe.DN in pipe_dns_lens.keys():
                pipe_dns_lens[pipe.DN] += pipe.length_km
            else:
                pipe_dns_lens[pipe.DN] = pipe.length_km
        all_pipes_len = [(dn, length) for dn, length in pipe_dns_lens.items()]

        # Get all demands
        demands_MW = [
            demand.flowrate_MW for demand in new_network.demand_nodes.values()
        ]

        # Get meter,ILI,valve costs
        meter_cost = bp_cost.meter_reg_station_cost(
            cp=self.costing_params, demands_MW=demands_MW
        )
        ili_costs = bp_cost.ili_cost(cp=self.costing_params, pipe_added=all_pipes_len)
        valve_cost = bp_cost.valve_replacement_cost(
            self.costing_params, all_pipes_len, self.ASME_params.location_class
        )

        # Run financial analysis to get LCOT
        price_breakdown = self.run_financial(
            new_pipe_cap=new_pipe_cap,
            comp_cost=comp_capex,
            revamped_comp_capex=revamped_comp_capex,
            supply_comp_capex=supply_comp_capex,
            all_fuel_MW=all_fuel_MW,
            all_fuel_elec_kW=all_fuel_elec_kW,
            supply_comp_fuel=supply_comp_fuel,
            cs_cost=cf_price,
            elec_cost=self.costing_params.elec_price,
            meter_cost=meter_cost,
            ili_costs=ili_costs,
            valve_cost=valve_cost,
            original_pipeline_cost=self.costing_params.original_pipeline_cost,
            financial_overrides=self.costing_params.financial_overrides,
        )

        # Print LCOT
        self.bp_print(
            f"LCOT: {price_breakdown['LCOT: Levelized cost of transport']} $/MMBTU"
        )

        # Make result files
        self.result_files(
            new_network,
            price_breakdown,
            cf_price,
            cap_cost,
            n_comps,
            l_comps,
            all_fuel_MW,
            all_fuel_elec_kW,
            all_new_pipes,
            revamped_comp_capex,
            meter_cost,
            valve_cost,
        )

        return price_breakdown["LCOT: Levelized cost of transport"]

    def run_financial(
        self,
        new_pipe_cap: float,
        comp_cost: list,
        revamped_comp_capex: list,
        supply_comp_capex: float,
        all_fuel_MW: float,
        all_fuel_elec_kW: float,
        supply_comp_fuel: dict,
        cs_cost: float,
        elec_cost: float,
        meter_cost: float,
        ili_costs: float,
        valve_cost: float,
        original_pipeline_cost: float,
        financial_overrides: float,
    ) -> pd.DataFrame:
        """
        Setup parameters to run LCOT calculation
        """

        # Use summation all of demands for capacity
        capacity = self.network.capacity_MMBTU_day

        # Get fuel usage rate
        all_fuel_usage = all_fuel_MW * gl.MW2MMBTUDAY / capacity  # MMBTU/MMBTU

        # Get fuel usage rate (electric)
        all_fuel_elec = all_fuel_elec_kW / capacity * gl.DAY2HR

        # Get fuel usage rate for supply compressor
        supply_comp_fuel["gas"] = supply_comp_fuel["gas"] * gl.MW2MMBTUDAY / capacity
        supply_comp_fuel["elec"] = supply_comp_fuel["elec"] / capacity * gl.DAY2HR

        # Get price breakdown
        price_breakdown = bp_cost.calc_lcot(
            json_file=self.casestudy_name,
            capacity=capacity,
            new_pipe_cap=new_pipe_cap,
            comp_cost=comp_cost,
            revamped_comp_capex=revamped_comp_capex,
            supply_comp_capex=supply_comp_capex,
            compressor_fuel=all_fuel_usage,
            compressor_fuel_elec=all_fuel_elec,
            supply_comp_fuel=supply_comp_fuel,
            cs_cost=cs_cost,
            elec_cost=elec_cost,
            meter_cost=meter_cost,
            ili_costs=ili_costs,
            valve_cost=valve_cost,
            original_network_residual_value=original_pipeline_cost,
            financial_overrides=financial_overrides,
        )

        return price_breakdown

    def result_files(
        self,
        new_network: BlendPATH_network,
        price_breakdown: pd.DataFrame,
        cs_cost: float,
        new_pipe: dict,
        n_comps: list,
        l_comps: list,
        all_fuel_MW: float,
        all_fuel_elec_kW: float,
        all_new_pipes: dict,
        revamped_comp_capex: list,
        meter_cost: float,
        valve_cost: float,
    ) -> None:
        """
        Create result file with summary values
        """
        results_file_name = f"{self.casestudy_name}/{self.results_dir}/ResultFiles/{self.mod_type.upper()}_{self.blend}_{self.design_option}{self.filename_suffix}.xlsx"

        # Specify output writer
        writer = pd.ExcelWriter(results_file_name, engine="xlsxwriter")
        workbook = writer.book

        #
        # Write results file - disclaimer
        # DISCLAIMER
        worksheet = workbook.add_worksheet("Disclaimer")
        startrow = 1
        worksheet.write_string(startrow, 0, disclaimer_message())
        # Wrap text
        worksheet.set_column("A:A", 100, workbook.add_format({"text_wrap": True}))

        #
        # Write results file - inputs
        # INPUTS
        worksheet = workbook.add_worksheet("Inputs")
        writer.sheets["Inputs"] = worksheet

        inputs = [
            ("Network name", self.casestudy_name),
            ("Save directory", self.results_dir),
            ("Design option - original", self.design_option),
            ("Location class", self.ASME_params.location_class),
            ("Joint factor", self.ASME_params.joint_factor),
            ("T de-rating factor", self.ASME_params.T_rating),
            ("Compression ratio", self.design_params.max_CR),
            ("Blending ratio", self.blend),
            ("Natural gas cost ($/MMBTU)", self.costing_params.ng_price),
            ("H2 cost ($/kg)", self.costing_params.h2_price),
            ("Electricity cost ($/kWh)", self.costing_params.elec_price),
            (
                "Final outlet pressure (MPa-g)",
                self.design_params.final_outlet_pressure_mpa_g,
            ),
            ("Region", self.costing_params.region),
            ("Modification method", self.mod_type),
            ("Modification design option", self.design_option_new),
            ("Verbose", self.verbose),
            ("Equation of state", self.eos),
            ("Inline inspection inspection interval", self.costing_params.ili_interval),
            (
                "Original pipeline depreciated cost",
                self.costing_params.original_pipeline_cost,
            ),
            (
                "Are added compressors electric?",
                self.design_params.new_comp_elec,
            ),
            (
                "Are existing compressors converted to electric?",
                self.design_params.existing_comp_elec,
            ),
            (
                "New gas compressor isentropic efficiency",
                self.design_params.new_comp_eta_s,
            ),
            (
                "New electric compressor isentropic efficiency",
                self.design_params.new_comp_eta_s_elec,
            ),
            (
                "New gas compressor driver efficiency",
                self.design_params.new_comp_eta_driver,
            ),
            (
                "New electric compressor driver efficiency",
                self.design_params.new_comp_eta_driver_elec,
            ),
        ]
        inputs = pd.DataFrame(inputs, columns=["Name", "Value"])
        startrow = 0
        inputs.to_excel(
            writer, sheet_name="Inputs", startrow=startrow, startcol=0, index=False
        )

        #
        # Write results file - results sheet
        # RESULTS
        worksheet = workbook.add_worksheet("Results")
        writer.sheets["Results"] = worksheet

        # Format compressors
        comp_breakdown = []
        comp_cost_total = 0
        comp_addl_rating = 0
        supply_comp = 0
        for comp in new_network.compressors.values():
            comp_breakdown.append(
                (
                    comp.name,
                    comp.shaft_power_MW,
                    comp.original_rating_MW,
                    comp.addl_rating,
                    comp.cost,
                    comp.revamp_cost,
                    comp.get_fuel_use_MMBTU_hr(),
                    comp.fuel_electric_W / gl.KW2W,
                )
            )
            comp_addl_rating += comp.addl_rating
            comp_cost_total += comp.cost
            if comp.name == "Supply compressor" and comp.original_rating_MW == 0:
                supply_comp = 1

        # Map nodes for determing pipe segments
        nodes_map = {}
        for ps_i, ps in enumerate(self.network.pipe_segments):
            nodes_map.update({x.name: ps_i for x in ps.nodes})

        # Fill in levelized costs
        params_out = []
        for i, v in price_breakdown.items():
            params_out.append((i, v, "$/MMBTU"))
        # Fill in other params
        params_out.append(
            ("Hydrogen injection price", self.costing_params.h2_price, "$/kg")
        )
        params_out.append(
            ("Natural gas price", self.costing_params.ng_price, "$/MMBTU")
        )
        if isinstance(self.costing_params.ng_price, dict) or isinstance(
            self.costing_params.h2_price, dict
        ):
            params_out.append(("Blended gas price", cs_cost, "$/MMBTU"))
        else:
            params_out.append(
                ("Blended gas price", cs_cost[min(cs_cost.keys())], "$/MMBTU")
            )
        params_out.append(
            ("Pipeline capacity (daily)", new_network.capacity_MMBTU_day, "MMBTU/day")
        )
        params_out.append(
            (
                "Pipeline capacity (hour)",
                new_network.capacity_MMBTU_day / gl.DAY2HR,
                "MMBTU/hr",
            )
        )
        pipeline_added_km = sum(new_pipe["length"])
        params_out.append(("Added pipeline", pipeline_added_km, "km"))
        params_out.append(("Added pipeline", pipeline_added_km * gl.KM2MI, "mi"))

        params_out.append(("Added compressor stations", sum(n_comps) + supply_comp, ""))
        params_out.append(("Added compressor capacity", comp_addl_rating, "hp"))
        params_out.append(
            ("Compressor fuel usage", all_fuel_MW * gl.MW2MMBTUDAY, "MMBTU/day")
        )
        params_out.append(
            (
                "Compressor fuel usage",
                all_fuel_MW * gl.MW2MMBTUDAY / gl.DAY2HR,
                "MMBTU/hr",
            )
        )
        params_out.append(
            (
                "Compressor fuel usage (electric)",
                all_fuel_elec_kW,
                "kW",
            )
        )
        params_out.append(
            (
                "New pipe",
                0 if not new_pipe["total cost"] else new_pipe["total cost"],
                "$",
            )
        )
        params_out.append(("New compressor stations", comp_cost_total, "$"))
        params_out.append(
            ("Compressor station refurbishment", sum(revamped_comp_capex), "$")
        )
        params_out.append(("Meter station modification", meter_cost, "$"))
        params_out.append(("Valve modifications", valve_cost, "$"))
        # params_out.append(("Original network residual value", 0, "$"))

        # Get energy ratio of H2
        pure_h2 = bp_plc.Composition({"H2": 1})
        GCV_H2_MJpsm3 = pure_h2.get_GCV()
        pure_ch4 = bp_plc.Composition(self.network.composition.pure_x)
        GCV_NG_MJpsm3 = pure_ch4.get_GCV()

        blend_ratio_energy = (self.blend * GCV_H2_MJpsm3) / (
            (self.blend * GCV_H2_MJpsm3) + (1 - self.blend) * GCV_NG_MJpsm3
        )
        params_out.append(("Hydrogen energy ratio", blend_ratio_energy * 100, "%"))

        params_out = pd.DataFrame(params_out, columns=["Parameter", "Value", "Units"])
        startrow = 0
        params_out.to_excel(
            writer, sheet_name="Results", startrow=startrow, startcol=0, index=False
        )
        startrow += params_out.shape[0] + 4

        # If mach number or erosional velocity is exceeded provide warning
        flag_mach = False
        flag_erosional = False
        for pipe in new_network.pipes.values():
            if pipe.get_mach_number() >= 1:
                flag_mach = True
            if pipe.get_erosional_velocity() <= max([pipe.v_from, pipe.v_to]):
                flag_erosional = True
        if flag_mach:
            red_text = workbook.add_format({"font_color": "red"})
            worksheet.write_string(
                startrow,
                0,
                "ERROR: Mach number exceeds 1, results may not be valid",
                red_text,
            )
            startrow += 1
        if flag_erosional:
            orange_text = workbook.add_format({"font_color": "orange"})
            worksheet.write_string(
                startrow,
                0,
                "WARNING: Pipeline gas velocities in pipeline exceed the ASME B31.12 para. I-3.4.5 erosional velocity. Consider changing BlendPATH input parameters to lower gas velocities or further investigate integrity risks associated with the high gas velocities in the modified pipeline design.",
                orange_text,
            )
            startrow += 1

        startrow += (flag_mach or flag_erosional) * 2

        #
        # Breakdown of original pipe
        # Loop thru pipes
        worksheet.write_string(
            startrow, 0, "Breakdown of original pipe by diameter, schedule and grade"
        )
        startrow += 1
        orig_pipes = {
            x: []
            for x in [
                "DSG",
                "Pipe Nominal Diameter",
                "Length (km)",
                "Grade",
                "Schedule",
            ]
        }
        for pipe in self.network.pipes.values():
            dsg = f"{pipe.DN}_{pipe.grade}_{pipe.schedule}"
            if dsg in orig_pipes["DSG"]:
                dsg_ind = orig_pipes["DSG"].index(dsg)
                orig_pipes["Length (km)"][dsg_ind] += pipe.length_km
            else:
                orig_pipes["DSG"].append(dsg)
                orig_pipes["Pipe Nominal Diameter"].append(pipe.DN)
                orig_pipes["Length (km)"].append(pipe.length_km)
                orig_pipes["Grade"].append(pipe.grade)
                orig_pipes["Schedule"].append(pipe.schedule)
        orig_pipes.pop("DSG")
        orig_pipes = pd.DataFrame(orig_pipes)
        orig_pipes.to_excel(
            writer, sheet_name="Results", startrow=startrow, startcol=0, index=False
        )
        startrow += orig_pipes.shape[0] + 4

        #
        # Breakdown of new pipe
        worksheet.write_string(
            startrow, 0, "Breakdown of new pipe by diameter, schedule and grade"
        )
        startrow += 1
        new_pipe_condensed = {
            x: []
            for x in [
                "Pipe Nominal Diameter",
                "Length (km)",
                "Steel Grade",
                "Schedule",
                "Material Cost ($)",
                "Labor Cost ($)",
                "Misc ($)",
                "ROW ($)",
            ]
        }
        if new_pipe["D_S_G"]:
            new_pipe_condensed["Pipe Nominal Diameter"] = new_pipe["DN"]
            new_pipe_condensed["Length (km)"] = new_pipe["length"]
            new_pipe_condensed["Steel Grade"] = new_pipe["grade"]
            new_pipe_condensed["Schedule"] = new_pipe["sch"]
            new_pipe_condensed["Material Cost ($)"] = new_pipe["mat_cost"]
            for i in new_pipe["anl_costs"]:
                new_pipe_condensed["Labor Cost ($)"].append(
                    i["Labor"] if "Labor" in i else 0
                )
                new_pipe_condensed["Misc ($)"].append(i["Misc"] if "Misc" in i else 0)
                new_pipe_condensed["ROW ($)"].append(i["ROW"] if "ROW" in i else 0)

        new_pipe_condensed = pd.DataFrame(new_pipe_condensed)
        new_pipe_condensed.to_excel(
            writer, sheet_name="Results", startrow=startrow, startcol=0, index=False
        )
        startrow += new_pipe_condensed.shape[0] + 4

        #
        # Breakdown of compressors
        worksheet.write_string(startrow, 0, "Breakdown of compressor by station")
        startrow += 1
        comp_breakdown = pd.DataFrame(
            comp_breakdown,
            columns=[
                "Station ID",
                "Shaft power (MW)",
                "Original Capacity (MW)",
                "Required Additional Capacity (hp)",
                "Cost (2020$)",
                "Revamp Cost (2020$)",
                "Fuel usage (MMBTU/hr)",
                "Electric power (kW)",
            ],
        )
        comp_breakdown.to_excel(
            writer, sheet_name="Results", startrow=startrow, startcol=0, index=False
        )
        startrow += comp_breakdown.shape[0] + 4

        # Pipes
        worksheet = workbook.add_worksheet("Modified network design")
        writer.sheets["Pipes profile"] = worksheet
        startrow = 0

        pipe_profile = []
        segment = 0
        for pipe in new_network.pipes.values():
            # Get pipe segment
            if pipe.from_node.name in nodes_map.keys():
                segment = nodes_map[pipe.from_node.name]
            elif pipe.to_node.name in nodes_map.keys():
                segment = nodes_map[pipe.to_node.name]

            # Get new/existing
            new_old = "Existing"
            maop = pipe.p_max_mpa_g
            if (
                self.mod_type in ["pl"]
                and pipe.name == all_new_pipes[f"pipe_segment_{segment}"]["name"]
            ):
                new_old = "New"
                maop = all_new_pipes[f"pipe_segment_{segment}"]["pressures"]
            if self.mod_type in ["dr"] and pipe.name in all_new_pipes.keys():
                new_old = "New"
                maop = all_new_pipes[pipe.name]["pressures"]

            pipe_profile.append(
                (
                    segment,
                    pipe.name,
                    pipe.from_node.name,
                    pipe.to_node.name,
                    new_old,
                    pipe.m_dot,
                    pipe.DN,
                    pipe.schedule,
                    pipe.thickness_mm,
                    pipe.grade,
                    maop,
                    pipe.length_km,
                    pipe.length_km * gl.KM2MI,
                    pipe.from_node.pressure,
                    pipe.to_node.pressure,
                    max([pipe.v_from, pipe.v_to]),
                    pipe.get_mach_number(),
                    pipe.get_erosional_velocity(),
                )
            )
        pipe_profile = pd.DataFrame(
            pipe_profile,
            columns=[
                "Pipe segment",
                "Pipe name",
                "FromName",
                "ToName",
                "Type",
                "Flow rate (kg/s)",
                "DN",
                "Schedule",
                "Thickness (mm)",
                "Steel grade",
                "MAOP (MPa-g)",
                "Length (km)",
                "Length (mi)",
                "Inlet pressure (Pa-g)",
                "Outlet pressure (Pa-g)",
                "Max velocity (m/s)",
                "Max Mach number",
                "Erosional velocity (m/s)",
            ],
        )
        pipe_profile.to_excel(
            writer,
            sheet_name="Pipes profile",
            startrow=startrow,
            startcol=0,
            index=False,
        )

        #
        # # New compressor breakdown - SHEET
        worksheet = workbook.add_worksheet("Compressor design")
        writer.sheets["Compressor design"] = worksheet
        startrow = 0
        ps_lengths = [0]
        existing_cs_lengths = {}
        for ps in self.network.pipe_segments:
            ps_lengths.append(ps.length_km + ps_lengths[-1])
            if ps.comps:
                existing_cs_lengths[ps.comps[0].name] = ps_lengths[-1]
        new_comp_breakdown = []
        comp_i = 0
        ps_i = 0
        for comp in new_network.compressors.values():
            segment_name = ""
            comp_type = "Existing"
            if comp.original_rating_MW == 0 and self.mod_type in [
                "ac",
                "additional_compressors",
            ]:
                comp_type = "New"
                if comp.name == "Supply compressor":
                    comp_length = 0
                else:
                    segment_name = ps_i
                    while not l_comps[ps_i]:
                        ps_i += 1
                    length = l_comps[ps_i][comp_i]
                    comp_length = length + ps_lengths[ps_i]
                    comp_i += 1
                    if comp_i == n_comps[ps_i]:
                        ps_i += 1
                        comp_i = 0

            elif comp.name in existing_cs_lengths.keys():
                comp_length = existing_cs_lengths[comp.name]
            else:
                comp_length = 0
            eta_s = comp.eta_comp_s if comp.fuel_extract else comp.eta_comp_s_elec
            eta_driver = (
                comp.eta_driver if comp.fuel_extract else comp.eta_driver_elec_used
            )
            new_comp_breakdown.append(
                (
                    segment_name,
                    comp.name,
                    comp.from_node.name,
                    comp.to_node.name,
                    comp_type,
                    comp_length,
                    comp_length * gl.KM2MI,
                    comp.get_cr_ratio(),
                    comp.get_fuel_use_MMBTU_hr(),
                    comp.shaft_power_MW,
                    comp.shaft_power_MW * gl.MW2HP,
                    comp.fuel_electric_W / gl.KW2W,
                    max([comp.shaft_power_MW, comp.original_rating_MW]),
                    eta_s,
                    eta_driver,
                    comp.cost,
                    comp.revamp_cost,
                    comp.cost + comp.revamp_cost,
                )
            )

        new_comp_breakdown = pd.DataFrame(
            new_comp_breakdown,
            columns=[
                "Segment",
                "Name",
                "FromName",
                "ToName",
                "Type",
                "Cumulative length (km)",
                "Cumulative length (mi)",
                "Pressure Ratio",
                "Fuel consumption (MMBTU/hr)",
                "Shaft power (MW)",
                "Shaft power (hp)",
                "Electric power (kW)",
                "Rating (MW)",
                "Isentropic efficiency",
                "Mechanical efficiency",
                "Cost ($)",
                "Revamp cost ($)",
                "Total cost ($)",
            ],
        )
        new_comp_breakdown.to_excel(
            writer,
            sheet_name="Compressor design",
            startrow=startrow,
            startcol=0,
            index=False,
        )

        # Pressure
        worksheet = workbook.add_worksheet("Pressure profile")
        writer.sheets["Pressure profile"] = worksheet
        startrow = 0
        pressure_breakdown = []
        for node in new_network.nodes.values():
            pressure_breakdown.append((node.name, node.pressure))
        pressure_breakdown = pd.DataFrame(
            pressure_breakdown, columns=["Node", "Pressure (Pa-g)"]
        )
        pressure_breakdown.to_excel(
            writer,
            sheet_name="Pressure profile",
            startrow=startrow,
            startcol=0,
            index=False,
        )

        ### Demand errors
        worksheet = workbook.add_worksheet("Demand error")
        writer.sheets["Demand error"] = worksheet
        startrow = 0
        demand_error = []
        for d_node in new_network.demand_nodes.values():
            hhv = d_node.node.heating_value()
            demand_error.append(
                (
                    d_node.name,
                    d_node.flowrate_mdot,
                    d_node.flowrate_mdot_sim,
                    hhv,
                    d_node.flowrate_mdot * hhv,
                    d_node.flowrate_mdot_sim * hhv,
                    (d_node.flowrate_mdot_sim - d_node.flowrate_mdot)
                    / d_node.flowrate_mdot
                    * 100,
                )
            )
        demand_error = pd.DataFrame(
            demand_error,
            columns=[
                "Demand node name",
                "Flow rate set point (kg/s)",
                "Flow rate calculated (kg/s)",
                "Higher heating value (MJ/kg)",
                "Energy set point (MW)",
                "Energy calculated (MW)",
                "Error in energy (%)",
            ],
        )
        demand_error.to_excel(
            writer,
            sheet_name="Demand error",
            startrow=startrow,
            startcol=0,
            index=False,
        )

        ### Closeout
        writer._save()

    def update_design_option(self, design_option: str, init: bool = False) -> None:
        """
        Update the design option of original pipeline

        Parameters:
        -----------
        design_option: str - Design option, must be 'a','b', or 'nfc'

        """
        self.design_option = bp_pa.check_design_option(design_option)

        # Reassess pipe with ASME B31.12
        self.network.pipe_assessment(
            self.ASME_params,
            design_option=self.design_option,
        )

        # Reassign segment ASME pressure, only if called after scenario is initialized
        if not init:
            self.bp_print(f"Updating existing pipe design option to: {design_option}")
            for ps in self.network.pipe_segments:
                ps.pressure_ASME_MPa = ps.pipes[0].pressure_ASME_MPa

    def blendH2(self, blend: float, h2_price: float = None, init: bool = False) -> None:
        """
        Blend amount of H2

        Parameters:
        -----------
        blend:float - Fraction of hydrogen
        h2_price: float = None - H2 price $/kg
        """
        # Check values
        try:
            blend = float(blend)
        except ValueError:
            raise ValueError(f"Blend must be numeric. The value entered was {blend}")
        if not (0 <= blend <= 1):
            raise ValueError(
                f"Blend must be between 0 and 1. The value entered was {blend}"
            )
        if not init:
            self.bp_print(f"Updating H2 blend to: {blend*100:0.2f}%")
        self.blend = blend
        self.network.blendH2(self.blend)

        # Check if new H2 price was specified and calculate new compressor fuel cost
        if h2_price is not None:
            self.costing_params.h2_price = h2_price
        self.costing_params.cf_price = bp_cost.get_cs_fuel_cost(
            blend,
            self.costing_params.ng_price,
            self.costing_params.h2_price,
            self.network.composition.pure_x,
            self.casestudy_name,
            self.costing_params.financial_overrides,
        )
        # Reset mass flow rates out of segments since changes to HHV will change mass flow rate
        self.network.pipe_segments[-1].mdot_out = self.network.pipe_segments[
            -1
        ].offtake_mdots[-1]

    def bp_print(self, msg: str) -> None:
        """
        Central location to print and check if verbose
        """
        if not self.verbose:
            return
        print(msg)
