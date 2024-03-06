from BlendPATH import *
import numpy as np

casestudy_name = 'Wangetal2018'
design_option_choice = 'no fracture control'
save_dir = casestudy_name + '/out/DO - ' + design_option_choice + '/'
blendpath = BlendPATH_scenario(casestudy_name = casestudy_name ,
                        save_directory = save_dir,
                        design_option = design_option_choice,
                        location_class  = 1,
                        joint_factor  = 1,
                        design_pressure_known = False,
                        pressure_drop_factor  = 1.0,
                        compression_ratio  = 1.55,
                        blending = 0.2,
                        natural_gas_cost = 7.39,
                        hydrogen_cost = 4.41,
                        nodes  = 50,
                        final_outlet_pressure=3325000,
                        region = 'GP',
                        PL_num_diams = 12,
                        original_CS_rating={'CS1':12.5,'CS2':12.5,'CS3':12.5},
                        inline_inspection_interval = 3
                        )

cost_overrides = {k:np.NaN for k in ['OVERRIDE: labor cost ($/in/mi)',
                                     'OVERRIDE: row cost ($/in/mi)',
                                     'OVERRIDE: misc cost ($/in/mi)',
                                     'OVERRIDE: Original pipeline cost ($)']}

blendpath.run_mod('direct_replacement','b')
#blendpath.run_mod('parallel_loop','b')
#blendpath.run_mod('additional_compressors','b')

LCOH = blendpath.run_financial(cost_overrides)

print(f'Levelized cost is: {round(LCOH,5)} $/MMBTU')

