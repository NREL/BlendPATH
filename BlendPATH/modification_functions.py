import BlendPATH.Design_assessment_functions as daf
import pandas as pd
import numpy as np
from scipy.optimize import fsolve,root
from math import *

import warnings
# This suppresses the fsolve warnings 
# warnings.filterwarnings('ignore', 'The iteration is not making good progress')  

def direct_replacement(yield_strengths,pipe_params,design_pressure,schedules,design_pressure_violations,steel_costs_per_kg,params,segment_props):
    '''
    Overview:
    ---------
        This function investigates replacing the pipe to satify constraints

    Parameters:
    -----------
        yield_strengths
        pipe_params
        design_pressure
        schedules
        design_pressure_violations
        steel_costs_per_kg
        params
        segment_props

    Returns:
    --------
        pipe_material_cost_direct_replacement

    '''
    #   Unload parameters in local variables
    design_option = params['design_option']
    location_class = params['location_class']
    joint_factor = params['joint_factor']
    density_steel = params['density_steel']
    
    # Run function to identify segments that need replacement if pressures fall below segment pressure constrains when
    # the MAOP is shifted for an existing pipe as per ASME B31.12
    excess_pressure_drop_violations = direct_replacement_excess_pressure_drop_violation(params, segment_props)

    # Compare excess_pressure_drop_violations with design_pressure_violation data frame
    # If a pipe segment show a high pressure (MAOP) violation but not an excess pressure drop violation, 
    # set value for pipe in Design Pressure Violation column in design_pressure_violation to 0 as
    # supply pressure/compressor outlet pressure could be shifted lower to ASME B31.12 existing pipe
    # design pressure if low pressure limits set in direct_replacement_excess_pressure_drop_violation 
    # are not exceeded

    for segment in excess_pressure_drop_violations.index:
        if excess_pressure_drop_violations["Excess pressure drop violation"][segment] == 0:
            design_pressure_violations.loc[(design_pressure_violations["Segment Name"] == segment),'Design Pressure Violation'] = 0

    # Run function to identify minimum viable schedule for each grade being considered 
    thickness_min,schedules_minviable,new_pressures = daf.schedules_minviable(design_option,location_class,joint_factor,\
                                                                    yield_strengths,pipe_params,design_pressure,\
                                                                    schedules,design_pressure_violations) 

    #-------------- Evaluate costs for direct replacement on a $/kg basis using Savoy Piping Inc. data ---------------   

    grades = yield_strengths['Grade']
    num_grades = yield_strengths.shape[0]

    numPipes_replacement = schedules_minviable.shape[0]

    pipes_to_replace = schedules_minviable.index.values


    #Pre-allocate dataframe for output
    pipe_material_cost_direct_replacement = pd.DataFrame(grades).set_index('Grade')
    thicknesses = pd.DataFrame(grades).set_index('Grade')
    

    for j in pipes_to_replace:
        pipe_cost_DR = []
        thickness=[]
        pick_row = pipe_params.loc[pipe_params['Pipe Name']==j]
        diameter_outer_DR = 0.001*pick_row['Outer Diameter [mm]'].iat[0]
        dn_DR = pick_row['DN'].iat[0]
        length_DR = 1000*pick_row['Length [km]'].iat[0]
        for k in grades:
            if pd.isna(schedules_minviable.loc[j,k]):
                pipe_cost_DR.append(np.nan)
                thickness.append(np.nan)
            else:
                schedule_k = schedules_minviable.loc[j,k]
                wall_thickness_DR = 0.001*schedules.loc[schedules['DN'] == dn_DR,schedule_k].iat[0]
                thickness.append(wall_thickness_DR)
                diameter_inner_DR = diameter_outer_DR - 2*wall_thickness_DR
                pipe_volume_DR = np.pi/4*(diameter_outer_DR**2 - diameter_inner_DR**2)*length_DR
                pipe_mass_DR = pipe_volume_DR*density_steel
                pipe_cost_perkg_DR = (steel_costs_per_kg.loc[steel_costs_per_kg['Grade'] == k,'Price [$/kg]']).values
                pipe_cost_perkg_DSR = pipe_cost_perkg_DR[0]
                pipe_cost_DR.append(pipe_cost_perkg_DR*pipe_mass_DR)
                
        pipe_cost_DR = pd.DataFrame(data = pipe_cost_DR)
        pipe_cost_DR = pipe_cost_DR.rename(columns = {pipe_cost_DR.columns[0]:j})
        pipe_cost_DR = pipe_cost_DR.join(grades).set_index('Grade')
        thickness = pd.DataFrame(data = thickness)
        thickness = thickness.rename(columns = {thickness.columns[0]:j})
        thickness = thickness.join(grades).set_index('Grade')
        
        pipe_material_cost_direct_replacement = pipe_material_cost_direct_replacement.join(pipe_cost_DR)
        thicknesses = thicknesses.join(thickness)
        
    pipe_material_cost_direct_replacement = pipe_material_cost_direct_replacement.T
    thicknesses = thicknesses.T 
    return pipe_material_cost_direct_replacement,schedules_minviable,thicknesses,new_pressures

def parallel_loop(node_fluidprops,segment_props,path,pipe_segments,yield_strengths,schedules,segment_demand_nodes,network_pipes,pipe_params,steel_costs_per_kg,params_in):
    '''
    Overview:
    ---------
        This function investigates placing a parallel loop to the pipe

    Parameters:
    -----------
        node_fluidprops
        segment_props
        path
        pipe_segments
        yield_strengths
        schedules
        segment_demand_nodes
        network_pipes
        pipe_params
        steel_costs_per_kg
        params_in
    
    Returns:
    --------
        length_loop_by_segment
        schedule_by_segment
        pipe_material_cost_PL_by_segment
    '''

    # Extract some parameters
    design_option = params_in['design_option']
    location_class = params_in['location_class']
    joint_factor = params_in['joint_factor']
    density_steel = params_in['density_steel']
    pressure_drop_factor = params_in['pressure_drop_factor']
    nodes = params_in['nodes']
    PL_num_diams = params_in['PL_num_diams']

    T_c = params_in['T_c']
    P_c = params_in['P_c']
    viscosity = params_in['viscosity']

    dn_options_all_segments = []

    
    def get_offtake_pressure(segment_props,inlet_flow_rate,inlet_pressure,inner_diameter,pressure_drop_factor,L_offtake,slope,nodes):
   
        m_dot_tot = inlet_flow_rate
        P_in = inlet_pressure                             # [Pa]
        D_1 = segment_props["Diameter [m]"]               # [m]
        D_2 = inner_diameter                              # [m]
        Z = segment_props["Z [-]"]                        # [-]
        R = segment_props["R [J/kg-K]"]                   # [J/kg-K]
        T = segment_props["Temperature [K]"]              # [K]
        fric_1 = segment_props["Friction factor [-]"]     # [-]
        fric_2 = segment_props["Friction factor [-]"]     # [-]
        RO = segment_props["Roughness [mm]"]/1000         # [m]
        params = (m_dot_tot,P_in,D_1,D_2,L_offtake,Z,R,T,fric_1,fric_2,pressure_drop_factor,nodes,T_c,P_c,RO,viscosity,slope)    
        
        # Establish initial guess matrix
        m_dot_1_init = m_dot_tot*0.5

        rho_guess = P_in/(Z*R*T)
        u_guess = m_dot_1_init/rho_guess/(np.pi*D_1**2/4)
        P_out_guess = P_in-0.5*rho_guess*u_guess**2*fric_1/D_1*L_offtake

        # Solve for outlet pressure
        p_1_init = np.linspace(P_in,P_out_guess,nodes)
        p_2_init = np.linspace(P_in,P_out_guess,nodes)
        x_init = np.concatenate((p_1_init[:-1],p_2_init[:-1],[m_dot_1_init,P_out_guess]))

        offtake_pressure_calc = root(parallelloop_getoutletpressure_knownlength_discretized,x_init,method='lm',args = params)

        return offtake_pressure_calc.x[-1]

    def get_min_loop_length(segment_props,inner_diameter,pressure_drop_factor,L_offtake,inlet_flow_rate,inlet_pressure,offtake_pressure_nodownstreamlooping,slope,nodes):

        m_dot_tot = inlet_flow_rate   
        P_1 = inlet_pressure                                        # [Pa]
        P_3 = offtake_pressure_nodownstreamlooping                  # [Pa]
        D_1 = segment_props["Diameter [m]"]                         # [m]
        D_2 = inner_diameter                                        # [m]
        D_3 = segment_props["Diameter [m]"]                         # [m]
        L_tot = L_offtake                                           # [m]
        Z = segment_props["Z [-]"]                                  # [-]
        R = segment_props["R [J/kg-K]"]                             # [J/kg-K]
        T = segment_props["Temperature [K]"]                        # [K]
        fric_1 = segment_props["Friction factor [-]"]               # [-]
        fric_2 = segment_props["Friction factor [-]"]               # [-]   
        fric_3 = segment_props["Friction factor [-]"]               # [-]  
        RO = segment_props["Roughness [mm]"]/1000                   # [m] 
        params = (m_dot_tot,P_1,P_3,D_1,D_2,D_3,L_tot,Z,R,T,fric_1,fric_2,fric_3,pressure_drop_factor,nodes,T_c,P_c,RO,viscosity,slope)

        # Establish initial guess matrix        
        m_dot_1_init = m_dot_tot*0.15
        L_1_init = L_tot*0.5

        rho_guess =P_1/(Z*R*T)
        u_guess = m_dot_1_init/rho_guess/(np.pi*D_1**2/4)
        P_out_guess = P_1-0.5*rho_guess*u_guess**2*fric_1/D_1*L_1_init

        p_1_init = np.linspace(P_1,P_out_guess,nodes)
        p_2_init = np.linspace(P_1,P_out_guess,nodes)
        p_3_init = np.linspace(P_out_guess,P_3,nodes)

        x_init = np.concatenate((p_1_init[:-1],p_2_init[:-1],p_3_init[:-1],[P_out_guess,L_1_init,m_dot_1_init]))

        out = root(parallelloop_getlength_knownpressure_discretized,x_init,method='lm',args = params)

        return out.x[-2]

    def get_inlet_pressure(m_dot_l,p_out_l,length_l,slope,segment_props,pressure_drop_factor,inlet_pressure_guess,nodes):
        params = (m_dot_l,p_out_l,segment_props["Diameter [m]"],length_l,\
            segment_props["Z [-]"],segment_props["R [J/kg-K]"],\
            segment_props["Temperature [K]"],segment_props["Friction factor [-]"],\
                pressure_drop_factor,nodes,T_c,P_c,segment_props["Roughness [mm]"]/1000,viscosity,slope)
            
        rho_guess = p_out_l/(segment_props["Z [-]"]*segment_props["R [J/kg-K]"]*segment_props["Temperature [K]"])
        u_guess = m_dot_l/rho_guess/(np.pi*segment_props["Diameter [m]"]**2/4)
        P_in_guess = p_out_l+0.5*rho_guess*u_guess**2*segment_props["Friction factor [-]"]/segment_props["Diameter [m]"]*length_l

        x_init = np.linspace(P_in_guess,p_out_l,nodes)

        out = root(singlepipe_getinletpressure_knownlength_discretized,x_init,method='lm',args = params)

        return out.x[0]


    grades = yield_strengths['Grade']

    # Initial output arrays per pipe segment
    length_loop_by_segment = []
    schedule_by_segment = []
    pipe_material_cost_PL_by_segment = []
    wall_thicknesses = []
    inner_diams = []
    #   Loop through all the pipe segments
    num_pipe_segments = len(pipe_segments)
    for j in range(num_pipe_segments):  

        #   Diameter options for parallel loop. Take first X options >= to segment diameter (default is 5)
        seg_DN = segment_props.loc[j,'Diameter [m]']*1000
        pick_DNs = schedules.loc[schedules['DN']>=seg_DN,'DN'].tolist()
        dn_options = pick_DNs[:PL_num_diams]
        dn_options_all_segments.append(dn_options)

        #   Retrieve ASME design pressure
        design_pressure_asme = segment_props.loc[j,'ASME design pressure [MPa]']  

        slope_j = segment_props.loc[j,'Elevation change [m]']/segment_props.loc[j,'Length [m]']
        
        #   Initialize output arrays by diameter
        schedule_by_dn_option = []
        length_loop_by_dn_option = []
        pipe_material_cost_PL_by_dn_option = []
        wall_thicknesses_dn = []
        inner_diam_dn = []
        #   Loop through diameters
        for dn in dn_options:
            #   Extract outer diameter from schedules
            diam_outer = schedules.loc[(schedules['DN']==dn),'Outer diameter [mm]'].iat[0]
            #   Extract viable schedules
            thickness_min,schedule_minviable = daf.schedules_minviable_generic(design_option,location_class,joint_factor,yield_strengths,dn,design_pressure_asme,schedules)
            schedule_minviable = schedule_minviable.dropna().reset_index(drop=True)

            #   Initialize output arrays by grade
            length_loop_by_grade = []
            pipe_material_cost_PL_by_grade = []
            wall_thickness_grade = []
            inner_diam_grade = []
            #   Loop through grades
            for i_grade,grade in enumerate(schedule_minviable['Grade']):
                #   Get geometry based on grade and the minimum viable schedule
                grade_minvi_schedule = schedule_minviable.loc[schedule_minviable['Grade']==grade,'Schedules'].values[0]
                wall_thickness_g = schedules.loc[schedules['DN']==dn,grade_minvi_schedule].values[0]
                inner_diameter = (diam_outer - 2*wall_thickness_g)/1000
                wall_thickness_grade.append(wall_thickness_g)
                inner_diam_grade.append(inner_diameter)
            
                #########################################################################
                #       STEP 1: Assess if 100% looping is adequate to meet demand       #
                #########################################################################
            
                # Establish length before/between each offtake
                demand_nodes_in_segment = segment_demand_nodes[j]
                pipes_in_segment = pipe_segments[j]
                pipesandnodes_in_segment = network_pipes.loc[network_pipes['Name'].isin(pipes_in_segment)].reset_index()
                last_node = pipesandnodes_in_segment['ToName'].iloc[-1]
            
                # If the last node in the segment is not a demand node, pretend that it is
                if len(demand_nodes_in_segment)==0 or ((len(demand_nodes_in_segment)>0) and demand_nodes_in_segment[-1] != last_node):
                    demand_nodes_in_segment.append(last_node)

                # Determine offtakes within segment and lengths between offtakes
                pipes_before_offtake = []
                length_before_offtake = []
                offtake_index = 0
                for demand_node in demand_nodes_in_segment:
                    offtake_index_new = pipesandnodes_in_segment.loc[pipesandnodes_in_segment['ToName']==demand_node,'Name'].index.values[0]+1
                    pipes_before_offtake_row = pipesandnodes_in_segment.iloc[offtake_index:offtake_index_new]
                    pipes_before_offtake.append(pipes_before_offtake_row['Name'].tolist())
                    length_before_offtake.append(pipe_params.loc[pipe_params['Pipe Name'].isin(pipes_before_offtake[-1]),'Length [km]'].sum()*1000)
                    offtake_index = offtake_index_new

                
                # Iterate through offtakes, determine if loop ends before/after offtake
                offtake_flowrate = node_fluidprops.loc[node_fluidprops['Node Name'].isin(demand_nodes_in_segment),['Node Name','Mass flow rate [kg/s]']].reset_index()
                inlet_pressure = segment_props.loc[j,"Inlet pressure [Pa]"]
                inlet_flow_rate = segment_props.loc[j,'Inlet Mass flow rate [kg/s]']
                length_sum = 0
                []
                for k in range(len(demand_nodes_in_segment)):
                    #   Skip if solution was generated from analysis of previous offtake
                    if len(length_loop_by_grade) == i_grade+1:
                        continue

                #########################################################################
                #       STEP 2: Check 100% looping before offtake and no looping after  #
                #########################################################################
                # First, check to see if 100% looping before offtake k followed by no looping
                # is adequate for meeting demand. If it is, then we know that the loop
                # ends before offtake k and we can simply calculate the precise loop length
                # based on the flow rate leading up to offtake k. If demand is not met,
                # then we need 100% looping before offtake k and should look at the segment before
                # the next offtake node. 
                    
                    # Calculate pressure at the offtake point assuming 100% looping
                    offtake_pressure_100pc_looping = get_offtake_pressure(segment_props.iloc[j],inlet_flow_rate,inlet_pressure,\
                        inner_diameter,pressure_drop_factor,length_before_offtake[k],slope_j,nodes) 
                    
                    # Calculate pressure at offtake k assuming no looping downstream of offtake k
                    if k == len(demand_nodes_in_segment)-1:
                        # If on the final offtake in the segment, then the 'offtake' pressure to compare against is just the segment outlet pressure
                        offtake_pressure_nodownstreamlooping = segment_props.loc[j,'Outlet pressure [Pa]']
                    else:
                        # Loop through the remaining offtake segments in reverse, calculating the inlet pressure to each
                        # Each segment, remove the offtake flow rate from the mass flow rate
                        p_out_l = segment_props.loc[j,'Outlet pressure [Pa]']
                        m_dot_l = segment_props.loc[j,'Outlet Mass flow rate [kg/s]']
                        length_l = length_before_offtake[k+1]
                        for l in reversed(range(k+1,len(demand_nodes_in_segment))):
                            inlet_pressure_l = get_inlet_pressure(m_dot_l,p_out_l,length_l,slope_j,segment_props.iloc[j],pressure_drop_factor,offtake_pressure_100pc_looping,nodes)
                            p_out_l = inlet_pressure_l
                            m_dot_l = m_dot_l - offtake_flowrate.loc[l-1,'Mass flow rate [kg/s]']
                            length_l = length_before_offtake[l-1]
                        
                        # The offtake pressure assuming no looping downstream of the offtake
                        offtake_pressure_nodownstreamlooping = inlet_pressure_l

                #########################################################################
                #       STEP 3: Check minimum loop length required                      #
                #########################################################################
                    
                    if offtake_pressure_nodownstreamlooping < offtake_pressure_100pc_looping:
                        # if the offtake pressure predicted from assuming no looping downstream
                        # of the offtake is less than the offtake pressure predicted from assuming
                        # 100% looping upstream of the offtake, then the 100% looping was excessive. 
                        # Next step is to calculate the minimum necessary loop length to achieve
                        # offtake_k_pressure_nodownstreamlooping

                        out = get_min_loop_length(segment_props.iloc[j],inner_diameter,\
                            pressure_drop_factor,length_before_offtake[k],inlet_flow_rate,inlet_pressure,offtake_pressure_nodownstreamlooping,slope_j,nodes)
                        # If min loop length is negative, then looping is not required.
                        if out < 0:
                            length_loop_by_grade.append(0)
                            continue
                        length_loop_by_grade.append(out+length_sum)
                    else:
                        # If the offtake pressure predicted from assuming no looping downstream of
                        # offtake k is greater than the offtake pressure predicted from assuming 100%
                        # looping up to offtake k, then you need that 100% looping up to offtake k
                        # and some more looping.
                        # Next step is to start over with the next loop
                        
                        if k == len(demand_nodes_in_segment)-1:
                            length_loop_by_grade.append(np.nan)
                        else:
                            inlet_pressure = offtake_pressure_100pc_looping
                            inlet_flow_rate = inlet_flow_rate + offtake_flowrate.loc[k,'Mass flow rate [kg/s]']
                            length_sum = length_sum+length_before_offtake[k]
                            length_remaining = segment_props.loc[j,'Length [m]'] - length_sum
                
                # Calculate pipe material cost
                pipe_volume_PL = np.pi/4*((diam_outer/1000)**2-inner_diameter**2)*(0 if len(length_loop_by_grade)==0 else length_loop_by_grade[i_grade])
                pipe_mass_PL = pipe_volume_PL*density_steel
                pipe_cost_perkg_PL = (steel_costs_per_kg.loc[steel_costs_per_kg['Grade']==grade,'Price [$/kg]']).iat[0]
                pipe_material_cost_PL_by_grade.append(pipe_cost_perkg_PL*pipe_mass_PL)
            
            # Convert outputs by grade to dataframe                   
            length_loop_by_grade = pd.DataFrame(length_loop_by_grade,columns = ['Loop length [m]'])
            length_loop_by_grade = length_loop_by_grade.join(grades)
            pipe_material_cost_PL_by_grade = pd.DataFrame(pipe_material_cost_PL_by_grade,columns = ['Pipe material costs [$]'])
            pipe_material_cost_PL_by_grade = pipe_material_cost_PL_by_grade.join(grades)

            # Aggregate outputs by grade into outputs by DN
            length_loop_by_dn_option.append(length_loop_by_grade)
            schedule_by_dn_option.append(schedule_minviable)
            pipe_material_cost_PL_by_dn_option.append(pipe_material_cost_PL_by_grade)
            wall_thicknesses_dn.append(wall_thickness_grade)
            inner_diam_dn.append(inner_diam_grade)
            
        
        # Aggregate outputs by DN to outputs by segment
        length_loop_by_segment.append(length_loop_by_dn_option)
        schedule_by_segment.append(schedule_by_dn_option)
        pipe_material_cost_PL_by_segment.append(pipe_material_cost_PL_by_dn_option)
        wall_thicknesses.append(wall_thicknesses_dn)
        inner_diams.append(inner_diam_dn)
    
    return length_loop_by_segment,schedule_by_segment,pipe_material_cost_PL_by_segment,dn_options_all_segments,wall_thicknesses,inner_diams

def additional_compressors(node_fluidprops,segment_props,pipe_segments,segment_demand_nodes,network_pipes,pipe_params,params):
    '''
    Overview:
    ---------
    This module looks at each individual pipe segment, segregated by existing compressors and changes in pipe diameter, 
    and, with an assumed MAOP and compressor compression ratio, calculates how many additional compressors would be 
    necessary to maintain a constant mass flow rate

    Parameters:
    -----------
        node_fluidprops
        segment_props
        pipe_segments
        segment_demand_nodes
        network_pipes
        pipe_params
        params

    Returns:
    --------
        num_compressors_by_segment
        compressor_length_by_segment
        compression_ratio_by_segment
    '''

    #   Initialize output arrays
    num_compressors_by_segment = []
    compressor_length_by_segment = []
    compression_ratio_by_segment = []
    pressures = []

    #   Unload parameters in local variables
    pressure_drop_factor = params['pressure_drop_factor']
    compression_ratio = params['compression_ratio']
    nodes = params['nodes']
    T_c = params['T_c']
    P_c = params['P_c']
    viscosity = params['viscosity']

    #   Determine number of pipe segments
    num_pipe_segments = len(pipe_segments)

    # Specify high initial length to default to length of subsegment for initialization
    length_after_comp = 1e9

    #   Loop thru pipe segments
    for j in range(num_pipe_segments):

        # Establish length before/between each offtake
        demand_nodes_in_segment = segment_demand_nodes[j]
        pipes_in_segment = pipe_segments[j]
        pipesandnodes_in_segment = network_pipes.loc[network_pipes['Name'].isin(pipes_in_segment)].reset_index()

        # If the last node in the segment is not a demand node, pretend that it is
        last_node = pipesandnodes_in_segment['ToName'].iloc[-1]
        if len(demand_nodes_in_segment)==0 or ((len(demand_nodes_in_segment)>0) and demand_nodes_in_segment[-1] != last_node):
            demand_nodes_in_segment.append(last_node)
        
        # Calculate the length of pipe between each offtake
        pipes_before_offtake = []
        length_before_offtake = []
        offtake_index = 0
        for demand_node in demand_nodes_in_segment:
            offtake_index_new = pipesandnodes_in_segment.loc[pipesandnodes_in_segment['ToName']==demand_node,'Name'].index.values[0]+1
            pipes_before_offtake_row = pipesandnodes_in_segment.iloc[offtake_index:offtake_index_new]
            pipes_before_offtake.append(pipes_before_offtake_row['Name'].tolist())
            length_before_offtake.append(pipe_params.loc[pipe_params['Pipe Name'].isin(pipes_before_offtake[-1]),'Length [km]'].sum()*1000)
            offtake_index = offtake_index_new

        # Establish offtake flowrates for segment j
        offtake_flowrate = node_fluidprops.loc[node_fluidprops['Node Name'].isin(demand_nodes_in_segment),['Node Name','Mass flow rate [kg/s]']].reset_index()
        
        # Setup lists for compressor number and placement for subsegment
        num_compressor_segment_j = 0
        length_compressor_segment_j = []
        compression_ratio_segment_j = []
        pressures_j = []
        m_dot_subseg = segment_props.loc[j,'Inlet Mass flow rate [kg/s]']
        # The maximum pressure after a compressor should be the maximum pressure
        # in the segment, which corresponds to that segment's design pressure
        p_high = segment_props.loc[j,'Inlet pressure [Pa]']
        # The lowest pressure we an go to is the max presure of the segment divided by the 
        # compression ratio of the new compressors
        p_low = p_high/compression_ratio

        length_remaining = 0
        num_skipped_subsegments = 0
        length_skipped_subsegments = []

        diam_j = segment_props.loc[j,"Diameter [m]"]
        Z_j = segment_props.loc[j,"Z [-]"]
        R_j = segment_props.loc[j,"R [J/kg-K]"]
        T_j = segment_props.loc[j,"Temperature [K]"]
        fric_j = segment_props.loc[j,"Friction factor [-]"]
        RO_j = segment_props.loc[j,"Roughness [mm]"]/1000 #m
        DH_j = segment_props.loc[j,"Elevation change [m]"]
        slope_j = DH_j/segment_props.loc[j,'Length [m]']

        

        for k in range(len(demand_nodes_in_segment)):
            if (j==num_pipe_segments-1) and (k == len(demand_nodes_in_segment)-1):
                p_low = segment_props.loc[j,'Outlet pressure [Pa]']
            length_subsegment = length_before_offtake[k]

            # Try to get initialization length in the ballpark
            length_init = min(length_after_comp,length_subsegment)
            
            # Calculate the length to the lowest pressure point 
            params = (m_dot_subseg,p_low,diam_j,p_high,Z_j,R_j,T_j,fric_j,pressure_drop_factor,nodes,T_c,P_c,RO_j,viscosity,slope_j)
            p_init = np.linspace(p_high,p_low,nodes-1) # P_in and P_out are known, so -1 for nodes
            x_init = np.concatenate(([length_init],p_init))
            length_after_comp_calc = fsolve(singlepipe_getlength_knownpressures_discretized,x_init,args = params)
            length_after_comp = length_after_comp_calc[0]
            
            if length_after_comp < length_subsegment:
                # If the length_after_comp is less than the length of the subsegment, then you need at least one compressor in this segment
                
                # Reset the list of skipped subsegments to zero cause we won't be skipping this one
                num_skipped_subsegments = 0
                
                # Add the first compressor to the list
                length_compressor_segment_j.append(round(length_after_comp+length_remaining))
                compression_ratio_segment_j.append(round(compression_ratio,3))
                pressures_j.append((segment_props.loc[j,'Inlet pressure [Pa]'],p_low))
                num_compressor_segment_j = num_compressor_segment_j+1
                
                
                # Because the subsegment might not start with a compressor, it is best to
                # calculate a new length_after_comp using the full potential compressor
                # pressure rise
                
                # Calculate remaining length after first compressor
                length_remaining = length_subsegment - length_after_comp
                
                # Calculate the length to the lowest pressure point considering full 
                # pressure range
                params = (m_dot_subseg,p_low,diam_j,segment_props.loc[j,'Inlet pressure [Pa]'],\
                    Z_j,R_j,T_j,fric_j,pressure_drop_factor,nodes,T_c,P_c,RO_j,viscosity,slope_j)             

                #p_init = np.linspace(p_high,p_low,nodes-1) # P_in and P_out are known, so -1 for nodes
                p_init = np.linspace(segment_props.loc[j,'Inlet pressure [Pa]'],p_low,nodes-1)
                x_init = np.concatenate(([length_subsegment],p_init))
                length_after_comp_calc = fsolve(singlepipe_getlength_knownpressures_discretized,x_init,args = params)
                length_after_comp = length_after_comp_calc[0]

                
                # Calculate the number of additional compressors needed in this subsegment
                num_comp_subsegment = floor(length_remaining/length_after_comp)
                length_compressor_segment_j.extend([round(length_after_comp)]*num_comp_subsegment)
                compression_ratio_segment_j.extend([round(compression_ratio,3)]*num_comp_subsegment)
                pressures_j.extend([(segment_props.loc[j,'Inlet pressure [Pa]'],p_low)]*num_comp_subsegment)

                # Add the new compressors to the tally
                num_compressor_segment_j = num_compressor_segment_j+num_comp_subsegment
                
                # Update length remaining with new compressors
                length_remaining = length_remaining - num_comp_subsegment*length_after_comp
                
                if k < len(demand_nodes_in_segment)-1:
                # For all but the last subsegment, calculate the pressure at the offtake 
                # to use as inlet pressure for the next subsegment
                    params = (m_dot_subseg,segment_props.loc[j,'Inlet pressure [Pa]'],diam_j,length_remaining,\
                        Z_j,R_j,T_j,fric_j,pressure_drop_factor,nodes,T_c,P_c,RO_j,viscosity,slope_j)
                
                    p_out_init = p_low
                    x_init = np.linspace(segment_props.loc[j,'Inlet pressure [Pa]'],p_out_init,nodes)
                    outlet_pressure_subseg_calc = fsolve(singlepipe_getoutletpressure_knownlength_discretized,x_init,args = params)
                    outlet_pressure_subseg = outlet_pressure_subseg_calc[-1] #Get last value (pressure at the end)


                    p_high = outlet_pressure_subseg
                    # Revise the flow rate for the next subsegment to account for the change from offtakes
                    m_dot_subseg = m_dot_subseg + offtake_flowrate.loc[k,'Mass flow rate [kg/s]']
                elif k == len(demand_nodes_in_segment)-1:
                # If this is the last subsegment, you should reduce the pressure ratio of the last compressor
                    
                    # The outlet pressure should be allowed to drop to the segment outlet pressure
                    outlet_pressure_subseg = segment_props.loc[j,'Outlet pressure [Pa]']
                    
                    # Calculate the inlet pressure given the known remaining length
                    params = (m_dot_subseg,outlet_pressure_subseg,segment_props.loc[j,"Diameter [m]"],length_remaining,\
                            segment_props.loc[j,"Z [-]"],segment_props.loc[j,"R [J/kg-K]"],\
                            segment_props.loc[j,"Temperature [K]"],segment_props.loc[j,"Friction factor [-]"],\
                            pressure_drop_factor,nodes,T_c,P_c,segment_props.loc[j,"Roughness [mm]"]/1000,viscosity,slope_j)
                        
                    p_in_init = p_high
                    x_init = np.linspace(p_high,outlet_pressure_subseg,nodes)
                    inlet_pressure_subseg_calc = fsolve(singlepipe_getinletpressure_knownlength_discretized,x_init,args = params)
                    inlet_pressure_subseg = inlet_pressure_subseg_calc[0] # Get first value of pressure array

                    # If the calculated compression ratio is greater than p_low,
                    if inlet_pressure_subseg > p_low:
                    
                        compression_ratio_segment_j[-1] = round(inlet_pressure_subseg/p_low,3)

                        pressures_j[-1] = (inlet_pressure_subseg,p_low)
                    
                    else:
                        # If the calculated inlet pressure is less than p_low, then that is probaly because
                        # the segment end pressure has been reduced. This means that the last compressor
                        # installed above was not actually necessary. 
                        compression_ratio_segment_j.pop(-1)
                        length_compressor_segment_j.pop(-1)
                        pressures_j.pop(-1)
                        num_compressor_segment_j = num_compressor_segment_j-1
                        
                   
                    p_high = outlet_pressure_subseg
                
            elif length_after_comp > length_subsegment:
                # If length_after_comp is greater than the subsegment length, then
                # you don't need a compressor in this subsegment. 
                
                # Add to counter of number of skipped subsegments (subsegments with no new compressor)
                num_skipped_subsegments = num_skipped_subsegments+1
                # Keep track of the length of skipped subsegments incase you get to the end of
                # the segment before you need another compressor
                if num_skipped_subsegments ==1:
                    if length_remaining > 0:
                        length_skipped_subsegments.append(length_remaining)
                    length_skipped_subsegments.append(length_subsegment)
                elif num_skipped_subsegments >1:
                    length_skipped_subsegments.append(length_subsegment)
                
                length_remaining = length_remaining+length_subsegment
                
                if k < len(demand_nodes_in_segment)-1:
                    # For all but the last subsegment, calculate the pressure at the 
                    # end of the subsegment to serve as inlet pressure for the next segment
                    params = (m_dot_subseg,p_high,diam_j,length_subsegment,\
                            Z_j,R_j,T_j,fric_j,pressure_drop_factor,nodes,T_c,P_c,RO_j,viscosity,slope_j)
                        
                    p_out_init = p_low
                    x_init = np.linspace(p_high,p_out_init,nodes)
                    outlet_pressure_subseg_calc = fsolve(singlepipe_getoutletpressure_knownlength_discretized,x_init,args = params)
                    outlet_pressure_subseg = outlet_pressure_subseg_calc[-1] #Get last value (pressure at the end)

                    p_high = outlet_pressure_subseg
                    # Revise the flow rate for the next subsegment to account for the change from offtakes
                    m_dot_subseg = m_dot_subseg + offtake_flowrate.loc[k,'Mass flow rate [kg/s]']
                    
                    
                elif k == len(demand_nodes_in_segment)-1:
                    # For the last subsegment, we need to calculate the reduced pressure ratio of
                    # the last newly installed compressor. 
                    
                    p_out_l = segment_props.loc[j,'Outlet pressure [Pa]']
                    m_dot_l = segment_props.loc[j,'Outlet Mass flow rate [kg/s]'] 
                    
                    # Loop through the 
                    for l in reversed(range(len(length_skipped_subsegments))):
                        
                        length_l = length_skipped_subsegments[l]
                        
                        params = (m_dot_l,p_out_l,diam_j,length_l,Z_j,R_j,T_j,fric_j,pressure_drop_factor,nodes,T_c,P_c,RO_j,viscosity,slope_j)
                            
                        p_in_init = segment_props.loc[j,'Inlet pressure [Pa]']
                        x_init = np.linspace(p_in_init,p_out_l,nodes)
                        inlet_pressure_subseg_calc = fsolve(singlepipe_getinletpressure_knownlength_discretized,x_init,args = params)
                        inlet_pressure_l = inlet_pressure_subseg_calc[0] # Get first value of pressure array

                        p_out_l = inlet_pressure_l
                        
                        offtake_flow_index = len(offtake_flowrate)-1-(len(length_skipped_subsegments) - l)
                        
                        if l >0:
                            m_dot_l = m_dot_l - offtake_flowrate.loc[offtake_flow_index,'Mass flow rate [kg/s]']

                    # After the loop
                    if len(compression_ratio_segment_j) > 0:
                        compression_ratio_segment_j[-1] = min(compression_ratio,round(inlet_pressure_l/p_low,3) )
                        if compression_ratio_segment_j[-1] < 1: # Becase of the low demand pressure, final compressor may not be required
                            compression_ratio_segment_j.pop(-1)
                            length_compressor_segment_j.pop(-1)
                            pressures_j.pop(-1)
                            num_compressor_segment_j = num_compressor_segment_j-1

        
        # Aggregate the number of additional compressors in each segment
        num_compressors_by_segment.append(num_compressor_segment_j)
        # Aggregate the length between new compressors in each segment. The first value gives the 
        # length from the beginning of the segment to the first additional compressor
        compressor_length_by_segment.append(length_compressor_segment_j)
        # Aggregate the compressor ratio of each additional compressor in the segment
        compression_ratio_by_segment.append(compression_ratio_segment_j)

        pressures.append(pressures_j)

    return num_compressors_by_segment,compressor_length_by_segment,compression_ratio_by_segment,pressures
    
#################################
# DIRECT REPLACEMENT
#################################

def direct_replacement_excess_pressure_drop_violation(params, segment_props):
    
    #   Load global parameters into local variables
    pressure_drop_factor = params['pressure_drop_factor']
    nodes = params['nodes']
    T_c = params['T_c']
    P_c = params['P_c']
    viscosity = params['viscosity']

    #   Set low limit on segment outlet pressures. Current limit is set to 20 bar which is around the inlet pressure of a city gates.
    #   Adjusting to a lower low limit may cause instability with fsolve as pressure difference (P2^2 - P1^2) becomes highly non-linear at low pressures
    p_out_lim = pd.DataFrame(2000000, index=np.arange(len(segment_props)), columns=["Low Pressure Limit [Pa]"])
    p_out_lim["Low Pressure Limit [Pa]"].iloc[-1] = segment_props["Outlet pressure [Pa]"].iloc[-1]

    #   Initialize output arrays
    segment_inlet_pressures = []

    #   Use singlepipe_getoutletpressure_knownlength_discretized to estimate inlet pressure at beginning of each segment
    #   given low limit on segment outlet pressures
    for j in segment_props.index:

        m_dot_j = segment_props.loc[j,"Inlet Mass flow rate [kg/s]"]
        p_out_j = p_out_lim.loc[j, "Low Pressure Limit [Pa]"]
        diam_j = segment_props.loc[j,"Diameter [m]"]
        length_segment = segment_props.loc[j,"Length [m]"]
        Z_j = segment_props.loc[j,"Z [-]"]
        R_j = segment_props.loc[j,"R [J/kg-K]"]
        T_j = segment_props.loc[j,"Temperature [K]"]
        fric_j = segment_props.loc[j,"Friction factor [-]"]
        RO_j = segment_props.loc[j,"Roughness [mm]"]/1000 #m
        DH_j = segment_props.loc[j,"Elevation change [m]"]
        slope_j = DH_j/segment_props.loc[j,'Length [m]']

        args = (m_dot_j,p_out_j,diam_j,length_segment,\
        Z_j,R_j,T_j,fric_j,pressure_drop_factor,nodes,T_c,P_c,RO_j,viscosity,slope_j)

        p_in_guess = (segment_props.loc[j,'Inlet pressure [Pa]'] - p_out_j) 
        x_init = np.linspace(p_in_guess, p_out_j, nodes)
        inlet_pressure_segment_j = fsolve(singlepipe_getinletpressure_knownlength_discretized,x_init,args = args)

        segment_inlet_pressures.append(inlet_pressure_segment_j[0])
    
    # Take modeled inlet pressure results and do a inlet pressure check to see if segment pressures
    # drop is excessive (calculated inlet pressure is higher than MAOP of existing pipe segment)

    # Initialize excess pressure drop violation dataframe
    excess_pressure_drop_violations = pd.DataFrame(0, index=np.arange(len(segment_props)), columns=["Inlet pressure violation"])
    excess_pressure_drop_violations = excess_pressure_drop_violations.join(pd.DataFrame(segment_inlet_pressures, columns=["Inlet pressure [Pa]"]))
    excess_pressure_drop_violations = excess_pressure_drop_violations.join(segment_props["ASME design pressure [MPa]"]*1000000)
    excess_pressure_drop_violations = excess_pressure_drop_violations.rename(columns = {"ASME design pressure [MPa]": 'ASME design pressure [Pa]'})

    # Identify segments that have pressures that drop below ambient or terminal demand node low pressure limit
    excess_pressure_drop_violations['Inlet Pressure Difference [Pa]'] = excess_pressure_drop_violations["ASME design pressure [Pa]"] - excess_pressure_drop_violations["Inlet pressure [Pa]"] 
    excess_pressure_drop_violations.loc[(excess_pressure_drop_violations['Inlet Pressure Difference [Pa]'] >=0),'Excess pressure drop violation'] = 0
    excess_pressure_drop_violations.loc[(excess_pressure_drop_violations['Inlet Pressure Difference [Pa]'] <0),'Excess pressure drop violation'] = 1

    return excess_pressure_drop_violations

#################################
# PARALLEL LOOP
#################################
def parallelloop_getlength_knownpressure_discretized(x,*params):
    '''
    Overview:
    ---------
    This function calculates parallel loop length for a given pipe segment. 
    The original pipe segment should have constant diameter and lie between two
    compression stations, such that the design pressure is uniform throughout 
    for linepacking up to the design pressure. The diameter of the parallel pipe
    is treated as a parameter but could be varied to explore the impact of parallel
    loop diameter on parallel loop length and overall cost.
    
    The system is divided into three pipes: pipe 1 is the original pipe upstream
    of the point at which the parallel loop re-intersects with the original pipe.
    Pipe 2 is the parallel loop; naturally pipes 1 and 2 have the same length.
    pipe 3 is the original pipe downstream of the point at which the parallel loop
    re-intersects with the original pipe. This system results in three thermodynamic
    statepoints: the pipe segment inlet, the pipe segment outlet, and the intermediate
    point where pipes 1, 2, and 3 intersect. Inlet and outlet pressure (and therefore
    density) are known, as are the pipe diameters, total flow rate, total length, gas
    compressibility, temperature, and friction factors. The model consists of basic
    conservation of mass and momentum, and estimates pressure drop through each pipe
    as a function of the average velocity within eahc pipe.

    Diagram:
    -----------
                                              ?L1?
                rho_in,v1_in---------------------------------------v1_out               L3
                                                                   > rho_int,P_int------------  v3,out,P_out,rho_out
                rho_in,v2_in---------------------------------------v2_out
                                              L2
    
    Parameters:
    -----------
    x: Array of guess values
        Order is P_1,P_2,P_3,P_int,L_1,m_dot_1
        P_1,P_2,P_3 are size of nodes-2
    params:
        0 - m_dot_tot
        1,2 - P_in,P_out
        3-5 - D1-D3
        6 - L_tot
        7-9 - Z,R,T
        10-12 - fric_1-fric_3
        13 - pressure_drop_factor
        14 - nodes
    
    Return:
    -------
        Array of residuals
    '''

    #Read in params
    m_dot_tot = params[0]                                           # Total mass flow rate [kg/s]
    P_in = params[1]                                                # Pipe segment inlet pressure [Pa]
    P_out = params[2]                                               # Pipe segment outlet pressure [Pa]
    D_1,D_2,D_3 = params[3:6]                                       # Pipe segment diameter [m] upstream of loop intersection
    L_tot = params[6]                                               # Total pipe segment length [m]
    R,T = params[8:10]                                              # Gas compressibility factor [-],Gas constant [J/kg-K],Gas constant [J/kg-K]
    fric_1,fric_2,fric_3 = params[10:13]                            # Friction factor for pipe 1
    pressure_drop_factor = params[13]                               # String that determines what density and velocity to use in pressure drop calc
    nodes = params[14]                                              # Number of nodes for discretization
    T_c = params[15]
    P_c = params[16]
    RO = params[17]
    viscosity = params[18]
    slope = params[19]

    # Calculate additional parameters
    A_1 = np.pi/4*D_1**2                                            # Cross-sectional area of pipe 1 [m2]
    A_2 = np.pi/4*D_2**2                                            # Cross-sectional area of pipe 2 [m2]
    A_3 = np.pi/4*D_3**2                                            # Cross-sectional area of pipe 3 [m2]
    
    # Read in local variable guess values
    N_less = nodes-1                                                # N_less is -1 nodes because inlet and outlet pressure is known
    P_1 = x[0:N_less]                                               # P_1 are the internal nodes in pipe 1 [Pa]
    P_2 = x[N_less:N_less*2]                                        # P_2 are the internal nodes in pipe 2 [Pa]
    P_3 = x[N_less*2:N_less*3]                                      # P_3 are the internal nodes in pipe 3 [Pa]
    P_int = x[-3]                                                   # P_int is outlet pressure of P_1 and P_2, and the inlet pressure to P_3 [Pa]
    L_1 = x[-2]                                                     # Length of pipe 1 [m]
    m_dot_1 = x[-1]                                                 # Mass flow rate into pipe 1 [kg/s]

    # Mass flow in each tube - from mass balance (m1+m2=m3=m_tot)
    m_dot_2 = m_dot_tot-m_dot_1                                     # Pipe 1 + Pipe 2 = total flow
    m_dot_3 = m_dot_tot                                             # Pipe 3 = total flow

    # Pressure [Pa]
    P_1_n = np.concatenate(([P_in],P_1,[P_int]))                    # Define pressures in pipe 1 with inlet (P_in ) and outlet (P_int) conditions [Pa]
    P_2_n = np.concatenate(([P_in],P_2,[P_int]))                    # Define pressures in pipe 2 with inlet (P_in ) and outlet (P_int) conditions [Pa]
    P_3_n = np.concatenate(([P_int],P_3,[P_out]))                   # Define pressures in pipe 3 with inlet (P_int) and outlet (P_out) conditions [Pa]

    #   Compressibility using Papay EOS
    P_r_1 = P_1_n/P_c
    P_r_2 = P_2_n/P_c
    P_r_3 = P_3_n/P_c
    T_r = T/T_c
    Z_1 = 1-(3.53*P_r_1/(10**(0.9813*T_r)))+(0.274*P_r_1**2/(10**(0.8157*T_r)))
    Z_2 = 1-(3.53*P_r_2/(10**(0.9813*T_r)))+(0.274*P_r_2**2/(10**(0.8157*T_r)))
    Z_3 = 1-(3.53*P_r_3/(10**(0.9813*T_r)))+(0.274*P_r_3**2/(10**(0.8157*T_r)))

    # Calculate length of pipes
    L_2 = L_1                                                       # L1 and L2 are same length [m]
    L_3 = L_tot-L_1                                                 # L1+L3 = L_tot [m]

    # Formulate momentum balances
    dz_1 = L_1/nodes                                                # Length of each node in pipe 1 [m]
    dz_2 = L_2/nodes                                                # Length of each node in pipe 2 [m]
    dz_3 = L_3/nodes                                                # Length of each node in pipe 3 [m]

    # Momentum balance components
    dpdz_1,tau_1,pududz_1,grav_1 = get_momentum_res(P_1_n,Z_1*R*T,A_1,dz_1,m_dot_1,D_1,RO,viscosity,slope) # Calculate momentum balance components for pipe 1
    dpdz_2,tau_2,pududz_2,grav_2 = get_momentum_res(P_2_n,Z_2*R*T,A_2,dz_2,m_dot_2,D_2,RO,viscosity,slope) # Calculate momentum balance components for pipe 2
    dpdz_3,tau_3,pududz_3,grav_3 = get_momentum_res(P_3_n,Z_3*R*T,A_3,dz_3,m_dot_3,D_3,RO,viscosity,slope) # Calculate momentum balance components for pipe 3

    # Momentum balance residual equations
    f_1 = -dpdz_1 - pressure_drop_factor*tau_1/D_1 - pududz_1 - grav_1       # Momentum balance for pipe 1 [Pa/m]
    f_2 = -dpdz_2 - pressure_drop_factor*tau_2/D_2 - pududz_2 - grav_2       # Momentum balance for pipe 2 [Pa/m]
    f_3 = -dpdz_3 - pressure_drop_factor*tau_3/D_3 - pududz_3 - grav_3       # Momentum balance for pipe 3 [Pa/m]
    
    return np.concatenate((f_1,f_2,f_3))

def parallelloop_getoutletpressure_knownlength_discretized(x,*params):
    '''
    Overview:
    ---------
    This function calculates the pressure at the end of a parllel loop of known
    length. Such a scenario would most likely occur when a parallel loop is known to
    extend beyond the length of a pipe that ends at an off-take or supply node. The diameter 
    of the parallel pipe is treated as a parameter but could be varied to explore the impact of parallel
    loop diameter on parallel loop length and overall cost.
    
    The system is divided into two pipes: pipe 1 is the original pipe between the segment
    inlet and the offtake or supply node. 
    Pipe 2 is the parallel loop with identical length to the original pipe. The 
    system has two thermodynamic statepoints: the inlet to the pipes and the outlet.
    The inlet pressure is known, but the outlet pressure is not. This function
    solves conservation of mass and momentum to calculate outlet pressure.

    Diagram:
    --------
                      v1_in---------------------------------------v1_out
     P_in,rho_in-----------                                       ------------  ?P_out?,rho_out
                      v2_in---------------------------------------v2_out
    Parameters:
    -----------
    x: Array of guess values
        Order is P_1,P_2,m_dot_1,P_out
        P_1,P_2 are size of nodes-2
    params:
        0 - m_dot_tot
        1 - P_in
        2-3 - D1-D2
        4 - L_tot
        5-7 - Z,R,T
        8-9 - fric_1-fric_2
        10 - pressure_drop_factor
        11 - nodes
    
    Return:
    -------
        Array of residuals


    '''

    
    #Read in params
    m_dot_tot = params[0]                                           # Total mass flow rate [kg/s]
    P_in = params[1]                                                # Pipe segment inlet pressure [Pa]
    D_1,D_2 = params[2:4]                                           # Original pipe segment diameter [m] upstream of loop intersection
    L_tot = params[4]                                               # Total pipe segment length [m]
    R,T = params[6:8]                                               # Gas compressibility factor [-], Gas constant [J/kg-K],Gas temperature [K]
    fric_1,fric_2 = params[8:10]                                    # Friction factor for pipe 1
    pressure_drop_factor = params[10]                               # String that determines what density and velocity to use in pressure drop calc
    nodes = params[11]                                              # Number of nodes for discretization
    T_c = params[12]
    P_c = params[13]
    RO = params[14]
    viscosity = params[15]
    slope = params[16]

    # Calculate cross section from pipe diameter [m2]
    A_1 = np.pi/4*D_1**2                                            # Cross-sectional area of pipe 1 [m2]
    A_2 = np.pi/4*D_2**2                                            # Cross-sectional area of pipe 2 [m2]

    # Read in local variable guess values
    N_less = nodes-1                                                # N_less is -1 nodes because solving for internal nodes
    P_1 = x[0:N_less]                                               # P_1 are the internal nodes in pipe 1 [Pa]
    P_2 = x[N_less:N_less*2]                                        # P_2 are the internal nodes in pipe 2 [Pa]
    m_dot_1 = x[-2]                                                 # Mass flow rate into pipe 1 [kg/s]
    P_out = x[-1]                                                   # Outlet pressure [Pa]
    
    # Mass balance
    m_dot_2 = m_dot_tot-m_dot_1                                     # Pipe 1 + Pipe 2 = total flow

    # Pressure [Pa]
    P_1_n = np.concatenate(([P_in],P_1,[P_out]))                    # Define pressures in pipe 1 with inlet (P_in) and outlet (P_out) conditions [Pa]
    P_2_n = np.concatenate(([P_in],P_2,[P_out]))                    # Define pressures in pipe 2 with inlet (P_in) and outlet (P_out) conditions [Pa]

    #   Compressibility using Papay EOS
    P_r_1 = P_1_n/P_c
    P_r_2 = P_2_n/P_c
    T_r = T/T_c
    Z_1 = 1-(3.53*P_r_1/(10**(0.9813*T_r)))+(0.274*P_r_1**2/(10**(0.8157*T_r)))
    Z_2 = 1-(3.53*P_r_2/(10**(0.9813*T_r)))+(0.274*P_r_2**2/(10**(0.8157*T_r)))

    dz_1 = L_tot/nodes                                              # Length of each node in pipe 1 [m]
    dz_2 = L_tot/nodes                                              # Length of each node in pipe 2 [m]

    dpdz_1,tau_1,pududz_1,grav_1 = get_momentum_res(P_1_n,Z_1*R*T,A_1,dz_1,m_dot_1,D_1,RO,viscosity,slope) # Calculate momentum balance components for pipe 1
    dpdz_2,tau_2,pududz_2,grav_2 = get_momentum_res(P_2_n,Z_2*R*T,A_2,dz_2,m_dot_2,D_2,RO,viscosity,slope) # Calculate momentum balance components for pipe 2
    
    f_1 = -dpdz_1 - pressure_drop_factor*tau_1/D_1 - pududz_1 - grav_1      # Momentum balance on pipe 1 [Pa/m]
    f_2 = -dpdz_2 - pressure_drop_factor*tau_2/D_2 - pududz_2 - grav_2      # Momentum balance on pipe 2 [Pa/m]

    return np.concatenate((f_1,f_2))


#################################
# SINGLE PIPE
#################################

def singlepipe_getoutletpressure_knownlength_discretized(x,*params):
    '''
    Overview:
    ---------
    This function calculates the pressure at the end of a single pipe of known
    length The system has two thermodynamic statepoints: the inlet and outlet.
    The inlet pressure is known, but the outlet pressure is not. This function
    solves conservation of mass and momentum to calculate outlet pressure.

    Diagram:
    --------
    P_in,v_in ---------------------------?P_out?,v_out

    Parameters:
    -----------
    x: Array of guess values
        Order is v_out,P_out
        v_out,P_out are size of nodes
    params:
        0 - m_dot_tot
        1 - P_in
        2 - D1
        3 - L_tot
        4-6 - Z,R,T
        7 - fric_1
        8 - pressure_drop_factor
        9 - nodes
    
    Return:
    -------
        Array of residuals

    '''

    #Read in params
    m_dot = params[0]                                   # Total mass flow rate [kg/s]
    P_in = params[1]                                    # Pipe segment inlet pressure [Pa]
    D_1 = params[2]                                     # Original pipe segment diameter [m] upstream of loop intersection
    L_tot = params[3]                                   # Total pipe segment length [m]
    R,T = params[5:7]                                   # Gas compressibility factor [-],Gas constant [J/kg-K],Gas temperature [K]
    fric = params[7]                                    # Friction factor for pipe 1
    pressure_drop_factor = params[8]                    # String that determines what density and velocity to use in pressure drop calc
    nodes = params[9]                                   # Number of nodes
    T_c = params[10]
    P_c = params[11]
    RO = params[12]
    viscosity = params[13]
    slope = params[14]

    # Calculate additional parameters
    A_c = np.pi/4*D_1**2                                # Cross-sectional area of pipe 1 [m2]   
    
    # Read in local variable guess values
    P = np.array(x)                                 # Fluid pressure at pipe outlet (from guess value) [Pa]

    #   Pressure [Pa]
    P_n = np.concatenate(([P_in],P))

    #   Compressibility using Papay EOS
    P_r = P_n/P_c
    T_r = T/T_c
    Z = 1-(3.53*P_r/(10**(0.9813*T_r)))+(0.274*P_r**2/(10**(0.8157*T_r)))

    dz = L_tot/nodes                                    # Discretization length (assume uniform)

    dpdz,tau,pududz,grav = get_momentum_res(P_n,Z*R*T,A_c,dz,m_dot,D_1,RO,viscosity,slope)
    
    # Residual Functions
    f_0 = -dpdz - pressure_drop_factor*tau/D_1 - pududz - grav # Momentum balance
    
    return f_0

def singlepipe_getinletpressure_knownlength_discretized(x,*params):
    '''
    Overview:
    ---------
    This function calculates the pressure at the inlet of a single pipe of known
    length The system has two thermodynamic statepoints: the inlet and outlet.
    The outlet pressure is known, but the inlet pressure is not. This function
    solves conservation of mass and momentum to calculate inlet pressure.

    Diagram:
    --------
          ?P_in?,v_in ---------------------------P_out,v_out

    Parameters:
    -----------
    x: Array of guess values
        Order is v_in,P_in
        v_in,P_in are size of nodes
    params:
        0 - m_dot_tot
        1 - P_out
        2 - D1
        3 - L_tot
        4-6 - Z,R,T
        7 - fric_1
        8 - pressure_drop_factor
        9 - nodes
    
    Return:
    -------
        Array of residuals    
    '''

    #Read in params
    m_dot = params[0]                               # Total mass flow rate [kg/s]
    P_out = params[1]                                   # Pipe segment inlet pressure [Pa]
    D_1 = params[2]                                     # Original pipe segment diameter [m] upstream of loop intersection
    L_tot = params[3]                                   # Total pipe segment length [m]
    R,T = params[5:7]                                 # Gas compressibility factor [-],Gas constant [J/kg-K],Gas temperature [K]
    fric = params[7]                                    # Friction factor for pipe 1
    pressure_drop_factor = params[8]                    # String that determines what density and velocity to use in pressure drop calc
    nodes = params[9]                                   #  Number of nodes
    T_c = params[10]
    P_c = params[11]
    RO = params[12]
    viscosity = params[13]
    slope = params[14]

    # Calculate additional parameters
    A_c = np.pi/4*D_1**2                                # Cross-sectional area of pipe 1 [m2]   
    
    # Read in local variable guess values
    P = np.array(x)                          # Fluid pressure at pipe inlet (from guess value) [Pa]
    P_n = np.concatenate((P,[P_out]))

    #   Compressibility using Papay EOS
    P_r = P_n/P_c
    T_r = T/T_c
    Z = 1-(3.53*P_r/(10**(0.9813*T_r)))+(0.274*P_r**2/(10**(0.8157*T_r)))
    
    dz = L_tot/nodes                                    # Discretization length (assume uniform) [m]
    
    dpdz,tau,pududz,grav = get_momentum_res(P_n,Z*R*T,A_c,dz,m_dot,D_1,RO,viscosity,slope)

    # Residual Functions
    f_0 = -dpdz - pressure_drop_factor*tau/D_1 - pududz - grav # Momentum balance

    return f_0

def singlepipe_getlength_knownpressures_discretized(x,*params):
    '''
    Overview:
    ---------
    This function calculates length of a single pipe given inlet and outlet pressures.
    The system has two thermodynamic statepoints: the inlet and outlet.
    This function solves conservation of mass and momentum. Note that in its current
    state, it does not actually require fsolve; however, future iterations that may
    involve discretization will and therefore it is beneficial to put this into a 
    functional form now.
    
    Diagram:
    --------
                                  ?L?
          P_in,v_in ---------------------------P_out,v_out

    Parameters:
    -----------
    x: Array of guess values
        Order is L_tot,P
        P is the size of nodes-1
    params:
        0 - m_dot_tot
        1 - P_out
        2 - D1
        3 - P_in
        4-6 - Z,R,T
        7 - fric_1
        8 - pressure_drop_factor
        9 - nodes
    
    Return:
    -------
        Array of residuals
    '''

    #Read in params
    m_dot = params[0]                               # Total mass flow rate [kg/s]
    P_out = params[1]                                   # Pipe segment inlet pressure [Pa]
    D_1 = params[2]                                     # Original pipe segment diameter [m] upstream of loop intersection
    P_in = params[3]                                    # Total pipe segment length [m]
    R,T = params[5:7]                                 # Gas compressibility factor [-],Gas constant [J/kg-K],Gas temperature [K]
    fric = params[7]                                  # Friction factor for pipe 1
    pressure_drop_factor = params[8]                    # String that determines what density and velocity to use in pressure drop calc
    nodes = params[9]                                   # Number of nodes
    T_c = params[10]
    P_c = params[11]
    RO = params[12]
    slope = params[13]
  
    viscosity = params[13]

    # Calculate additional parameters
    A_c = np.pi/4*D_1**2                                # Cross-sectional area of pipe 1 [m2]   
    
    # Read in local variable guess values
    L_tot = x[0]                                        # Outlet fluid velocity (from guess value) [m/s]
    P = np.array(x[1:])                                 # Fluid pressure at pipe outlet (from guess value) [Pa]
    P_n = np.concatenate(([P_in],P,[P_out]))

    #   Compressibility using Papay EOS
    P_r = P_n/P_c
    T_r = T/T_c
    Z = 1-(3.53*P_r/(10**(0.9813*T_r)))+(0.274*P_r**2/(10**(0.8157*T_r)))
    
    dz = L_tot/nodes                                    # Discretization length (assume uniform)

    dpdz,tau,pududz,grav = get_momentum_res(P_n,Z*R*T,A_c,dz,m_dot,D_1,RO,viscosity,slope)
    
    # Residual Functions
    f_0 = -dpdz - pressure_drop_factor*tau/D_1 - pududz - grav # Momentum balance

    return f_0

def get_momentum_res(p,zrt,A,dz,m_dot,D,RO,viscosity,slope):
    '''
    Overview:
    ---------
        This funciton calculates the terms of the momentum balance

    Parameters:
    -----------
        p : list[float] - Array of pressures
        zrt : float - Product of Z R and T
        A : float - Cross sectional area of pipe
        dz : float - Length of discretized node
        m_dot : floar - Mass flow through pipe
        fric : Darcy friction factor

    Returns:
    --------
        dpdz : list[float] - Array of pressure drops for each node
        tau : list[float] - Array of viscous drag terms
        pududz : list[float] - Array of velocity change terms
    '''
    rho = p/zrt                                         # Density of gas [kg/m3]
    rho_avg = np.mean([rho[0:-1],rho[1:]],axis=0)       # Average density in each node
    v = m_dot/rho/A                                     # Velocity of gas[m/s]
    v_avg = np.mean([v[0:-1],v[1:]],axis=0)             # Average density in each node

    #   Friction factor from Hofer correlation
    Re = rho_avg*v_avg*D/viscosity
    f = (-2*np.log10(4.518/Re*np.log10(Re/7)+RO/(3.71*D)))**(-2)

    g = 9.81 #m/s2
    
    dpdz = (p[1:]-p[0:-1])/dz                           # Pressure drop
    tau = 0.5*f*rho_avg*v_avg**2                        # Viscous drag
    pududz = rho_avg*v_avg * (v[1:]-v[0:-1])/dz         # Inertia term
    grav = rho_avg*g*slope

    return dpdz,tau,pududz,grav 