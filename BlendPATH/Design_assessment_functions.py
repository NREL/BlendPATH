# -*- coding: utf-8 -*-
"""
Functions for HyBlend Pipeline Prep Tool Design Assessment Module
These function are primarily for retriving SAInt data 
or length/pressure of a single or parallel pipe 
"""

from ctypes import *
from math import *
import pandas as pd
import numpy as np
import cantera as ct
from pkg_resources import resource_filename

from scipy.optimize import fsolve,root
import BlendPATH.modification_functions as mod_fxn


# Define function for running SAInt Scenario
def runSAInt(mydll,path_to_files,suffix):

    # Load .xlsx network file
    # mydll.importGNET(casestudy_path + casestudy_name + "_Network.xlsx")
    mydll.importGNET(f"{path_to_files}_Network{suffix}.xlsx")

    # Create Scenario File
    mydll.newGSCE("STE", "SteadyGas", "01/01/2022 0:00", "01/01/2022 1:00", 3600)

    # Load .xlsx scenario file
    mydll.importGSCE(f"{path_to_files}_Event{suffix}.xlsx")

    # Run Simulation
    a = mydll.runGSIM()


    if a==64:
        status = 'Failed'
        print(status)
    else:
        status = 'Success'

    return mydll, status

# Define function for retrieving node pressure.
def getnodePressure(dllfolder,casestudy_name):
    casestudy_path = 'Case Studies\\' + casestudy_name + '\\' 
    # Load SAInt API into local space
    mydll = cdll.LoadLibrary(dllfolder + "SAInt-API.dll")
    mydll.evalStr.restype=c_wchar_p
    mydll.evalInt.restype=c_int
    mydll.evalBool.restype=c_bool
    mydll.evalFloat.restype=c_float
    
    # Get number of nodes in network for iterating
    numNO = mydll.evalInt("GNET.NUMNO")
    
    # Get pipe pressures at nodes and put into a dataframe
    node_name = []
    node_pressure = []
    for j in range(numNO):
        node_name.append(mydll.evalStr('GNO.{0}.Name'.format(j)))
        node_pressure.append(mydll.evalFloat('GNO.{0}.P'.format(j)))
    node_name = pd.DataFrame(node_name,columns = ['Node Name'])
    node_pressure = node_name.join(pd.DataFrame(node_pressure,columns = ['Pressure [bar-g]']))
            
    return(node_pressure)

# Define function for retrieving node pressure.
def getnodefluidprops(mydll,casestudy_name):
    
    # Get number of nodes in network for iterating
    numNO = mydll.evalInt("GNET.NUMNO")
    
    # Get pipe pressures at nodes and put into a dataframe
    node_fluidprops_names = {'Node Name':'Name','Pressure [bar-g]':'P','Temperature [C]':'T','Molar Mass [g/mol]':'NQ!M','Density [kg/m3]':'RHO',\
                           'Gas Constant [J/kg-K]':'NQ!R','Gross Calorific Value [MJ/sm3]':'NQ!GCV','Vol flow [m3/s]':'QVOL','Thermal Flow [MW]':'TQ',\
                            'Mass flow rate [kg/s]':'Q','Height [m]':'H[m]'}
    node_fluidprops_dict = {key:[] for key in node_fluidprops_names.keys()}

    for j in range(numNO):
        for i,v in node_fluidprops_names.items():
            if i == 'Node Name':
                node_fluidprops_dict[i].append(mydll.evalStr(f'GNO.{j}.{v}'))
                continue
            if i == 'Mass flow rate [kg/s]':
                node_fluidprops_dict[i].append(mydll.evalFloat(f'GNO.{j}.{v}.[kg/s]'))
                continue
            node_fluidprops_dict[i].append(mydll.evalFloat(f'GNO.{j}.{v}'))
    node_fluidprops = pd.DataFrame(node_fluidprops_dict)
            
    return(node_fluidprops)

# Define function for retrieving pipe parameters (geometry and hydraulic properties)
def getpipeParams(mydll,numPipes):

    # Get pipe geometric and hydraulic properties and put into dataframe
    pipe_params_names = {'Pipe Name':'Name','Pipe Diameter [mm]':'D','Length [km]':'L','Wall thickness [mm]':'WTH','Max pressure [bar-g]':'P',\
                           'Max velocity [m/s]':'V','Roughness [mm]':'RO','Outer Diameter [mm]':'DOUT'}
    pipe_params_dict = {key:[] for key in pipe_params_names.keys()}

    for j in range(numPipes):
        for i,v in pipe_params_names.items():
            if i == 'Pipe Name':
                pipe_params_dict[i].append(mydll.evalStr(f'GPI.{j}.{v}'))
                continue
            if i in ['Max pressure [bar-g]','Max velocity [m/s]']:
                val_in = mydll.evalFloat(f'GPI.{j}.{v}I')
                val_out = mydll.evalFloat(f'GPI.{j}.{v}O')
                pipe_params_dict[i].append(max(val_in,val_out))
                continue
            pipe_params_dict[i].append(mydll.evalFloat(f'GPI.{j}.{v}'))
    pipe_params = pd.DataFrame(pipe_params_dict)
    pipe_params['Max pressure [MPa]'] = pipe_params['Max pressure [bar-g]']/10
    
    return(pipe_params)

# Define function for retrieving fluid properties
def getpipefluidprops(mydll,numPipes):
    
    # Get pipe geometric and hydraulic properties and put into dataframe
    fluidprops_names = {'Pipe Name':'Name','Inlet Pressure [bar-g]':'PI','Outlet Pressure [bar-g]':'PO','Temperature [C]':'T','Inlet Density [kg/m3]':'RHOI',\
                           'Outlet Density [kg/m3]':'RHOO','Inlet Velocity [m/s]':'VI','Outlet Velocity [m/s]':'VO','Inlet Vol Flow [m3/s]':'QVOLI',\
                            'Outlet Vol Flow [m3/s]':'QVOLO','Reynolds Number [-]':'REY','Friction Factor [-]':'LAM','Compressibility Factor [-]':'Z',\
                            'Roughness [mm]':'RO','Mass flow rate [kg/s]':'Q','Elevation change [m]':'DH[m]'}
    fluidprops_dict = {key:[] for key in fluidprops_names.keys()}

    for j in range(numPipes):
        for i,v in fluidprops_names.items():
            if i == 'Pipe Name':
                fluidprops_dict[i].append(mydll.evalStr(f'GPI.{j}.{v}'))
                continue
            if i == 'Mass flow rate [kg/s]':
                fluidprops_dict[i].append(mydll.evalFloat(f'GPI.{j}.{v}.[kg/s]'))
                continue
            fluidprops_dict[i].append(mydll.evalFloat(f'GPI.{j}.{v}'))
    fluidprops = pd.DataFrame(fluidprops_dict)

    return(fluidprops)

# Define function to calculate design pressure using ASME B31.12
def designpressure_ASMEB3112(design_option,location_class,joint_factor,spec_min_yield_strength,spec_min_tens_strength,design_pressure,diameter,wall_thickness):
    # Design factor table based on option. Could also make option a the default and 
    # use boolean to allow user to decide if they want to use option B
    if design_option.lower() == 'a':
        design_factor_list = [0.5,0.5,0.5,0.4]
    elif design_option.lower() == 'b':
        design_factor_list = [0.72,0.60,0.50,0.40]
    elif design_option.lower() == 'no fracture control':
        design_factor_list = [0.4,0.4,0.4,0.4]
    else:
        design_factor_list = [0.4,0.4,0.4,0.4]
        print("Error. Invalid design option. Option set to 'No fracture control' as default. Please enter a, A, b, B, or 'No fracture control' for design option")
    
    # Determine design factor based on location class
    design_factor = design_factor_list[location_class - 1]
    
    #Array of design pressures for ASME B31.12 Table IX-5A in MPa
    dp_array = np.array([6.8948,13.7895,15.685,16.5474,17.9264,19.3053,20.6843])
    
    # Create arrays of material performance factors based on SMYS and SMTS for ASME
    # B31.12 Table IX-5A for stresses and design pressure in MPa
    h_f_array = ASME_hf(spec_min_yield_strength,spec_min_tens_strength)   
    
    # Interpolate ASME B31.12 Table IX-5A
    if design_option.lower() in ['b','no fracture control']:
        mat_perf_factor = 1
    else:
        mat_perf_factor = np.interp(design_pressure,dp_array,h_f_array) 
    # Calculate revised design pressure based on material performance factor
    design_pressure_revised = 2*spec_min_yield_strength*wall_thickness/(diameter)*design_factor*joint_factor*mat_perf_factor
    # Calculate error between revised design pressure and assigned design pressure
    error = abs(design_pressure - design_pressure_revised)/design_pressure_revised*100
    # Loop through material performance factor interpolation and design pressure calculation until they converge
    while error >0.001: 
        design_pressure = design_pressure_revised
        #mat_perf_factor = np.interp(design_pressure,dp_array,h_f_array)
        if design_option.lower() in ['b','no fracture control']:
            mat_perf_factor = 1
        else:
            mat_perf_factor = np.interp(design_pressure,dp_array,h_f_array)   
        design_pressure_revised = 2*spec_min_yield_strength*wall_thickness/(diameter)*design_factor*joint_factor*mat_perf_factor
        error = abs(design_pressure - design_pressure_revised)/design_pressure_revised*100
        
    return(design_pressure_revised)

# Define function to identify minimum thickness and thus schedule for equivalent diameter with different materials
def schedules_minviable(design_option,location_class, joint_factor,yield_strengths,pipe_params,design_pressure,schedules,design_pressure_violations):
    
    # Identify grades being considered and count
    grades = yield_strengths['Grade']
    number_grades = yield_strengths.shape[0]
    
    #Count number of pipes
    numPipes = pipe_params.shape[0]
    
    # Pre-allocate dataframe for output
    schedule_minviable_allpipes = pd.DataFrame(grades).set_index('Grade')
    thickness_min_allpipes = pd.DataFrame(grades).set_index('Grade')
    pressures_allpipes = pd.DataFrame(grades).set_index('Grade')
            
    for j in range(numPipes):
        
        # Assess modifications only if design pressure violations are seen
        if design_pressure_violations.loc[j,'Design Pressure Violation'] == 1:
           
            # Clean up table leaving what is actually needed
            schedules_DN = schedules.loc[(schedules['DN'] == pipe_params.loc[j,'DN'])].drop(labels = ['DN','Outer diameter [mm]'], axis = 1).transpose().reset_index()
            schedules_DN = schedules_DN.rename(columns = {schedules_DN.columns[0]:'Schedule',schedules_DN.columns[1]:'Wall thickness [mm]'}).dropna()
            
            diam_DN = pipe_params.loc[j,'DN']
            # For a given pipe, loop through grades and evaluate minimum thickness and minimum viable grade
            thickness,schedule_minviable,new_pressure = ASME_design(number_grades,yield_strengths,design_option,design_pressure.loc[j],diam_DN,location_class,joint_factor,schedules_DN)
               
            # Convert thickness to dataframe
            thickness_min = pd.DataFrame(data = thickness,columns=['Schedules'])
            thickness_min = thickness_min.join(grades)
            
            # Get rid of extra schedules that have the same thickness
            for g in range(grades.shape[0]):
                schedule_minviable[g] = schedule_minviable[g][0]
            
            # Convert min viable grade to dataframe and set up for joining to dataframe of min viable schedules of other pipes
            schedule_minviable = pd.DataFrame(data = schedule_minviable,columns=['Schedules'])
            schedule_minviable = schedule_minviable.join(grades)
           
            thickness_min = thickness_min.explode('Schedules')
            thickness_min = thickness_min.rename(columns = {'Schedules':design_pressure_violations.loc[j,'Pipe Name']}).set_index('Grade')
           
            # For some reason this is necessary to format things correclty
            schedule_minviable = schedule_minviable.explode('Schedules')
            # Rename column header and set index to grade for easy identification later and transposition
            schedule_minviable = schedule_minviable.rename(columns = {'Schedules':design_pressure_violations.loc[j,'Pipe Name']}).set_index('Grade')

            pressures=pd.DataFrame(data=new_pressure,columns=['Schedules']).join(grades).rename(columns = {'Schedules':design_pressure_violations.loc[j,'Pipe Name']}).set_index('Grade')
            

            # Join with previously evaluated pipes
            thickness_min_allpipes = thickness_min_allpipes.join(thickness_min)
            schedule_minviable_allpipes = schedule_minviable_allpipes.join(schedule_minviable)
            pressures_allpipes = pressures_allpipes.join(pressures)
        
    # Transpose the entire output    
    thickness_min_allpipes = thickness_min_allpipes.T
    schedule_minviable_allpipes = schedule_minviable_allpipes.T
    pressures_allpipes = pressures_allpipes.T
     
    return(thickness_min_allpipes,schedule_minviable_allpipes,pressures_allpipes)

# Define function to identify minimum thickness and thus schedule for equivalent diameter with different materials
def schedules_minviable_generic(design_option,location_class,joint_factor,yield_strengths,dn,design_pressure_asme,schedules):
    
    # Identify grades being considered and count
    grades = yield_strengths['Grade']
    number_grades = yield_strengths.shape[0]
       
    # Clean up table leaving what is actually needed
    schedules_DN = schedules.loc[(schedules['DN'] == dn)].drop(labels = ['DN','Outer diameter [mm]'], axis = 1).transpose().reset_index()
    schedules_DN = schedules_DN.rename(columns = {schedules_DN.columns[0]:'Schedule',schedules_DN.columns[1]:'Wall thickness [mm]'}).dropna()
    schedules_DN = schedules_DN.reset_index().drop(labels = ['index'],axis=1)

    diam_outer = schedules.loc[(schedules['DN']==dn),'Outer diameter [mm]'].values
    
    # For a given pipe, loop through grades and evaluate minimum thickness and minimum viable grade
    thickness,schedule_minviable,_ = ASME_design(number_grades,yield_strengths,design_option,design_pressure_asme,diam_outer[0],location_class,joint_factor,schedules_DN)
    
    # Convert thickness to dataframe
    thickness_min = pd.DataFrame(data = thickness)
    thickness_min = thickness_min.rename(columns = {thickness_min.columns[0]:'Thickness [mm]'})
    thickness_min = thickness_min.join(grades)#.set_index('Grade')
    
    # Get rid of any duplicate schedules
    for g in range(grades.shape[0]):
        schedule_minviable[g] = schedule_minviable[g][0]
    # Convert min viable grade to dataframe and set up for joining to dataframe of min viable schedules of other pipes
    schedule_minviable = pd.DataFrame(data = schedule_minviable)
    schedule_minviable = schedule_minviable.rename(columns = {schedule_minviable.columns[0]:'Schedules'})
    schedule_minviable = schedule_minviable.join(grades)#.set_index('Grade')
    
    return(thickness_min,schedule_minviable)

def ASME_design(number_grades,yield_strengths,design_option,pressure,OD,location_class,joint_factor,schedules_DN):
    
    
    # Assign design factor based on design option
    if design_option.lower() == 'a':
        design_factor_list = [0.5,0.5,0.5,0.4]
    elif design_option.lower() == 'b':
        design_factor_list = [0.72,0.60,0.50,0.40]
    else:
        design_factor_list = [0.5,0.5,0.5,0.4]
        print('Error. Invalid design option. Option set to a as default. Please enter a, A, b, or B for design option')
    
    # Determine design factor based on location class
    design_factor = design_factor_list[location_class - 1]
    
    # Design pressure array for ASME B31.12 Table for material performance factor
    dp_array = np.array([6.8948,13.7895,15.685,16.5474,17.9264,19.3053,20.6843])

    thickness = []
    schedule_minviable = []
    new_pressure = []
    for k in range(number_grades):
        if design_option.lower() in ['b','no fracture control']:
            mat_perf_factor = 1
        else: 
            h_f_array = ASME_hf(yield_strengths.loc[k,'SMYS [Mpa]'],yield_strengths.loc[k,'SMTS [Mpa]']) 
            mat_perf_factor = np.interp(pressure,dp_array,h_f_array)
        thickness_k = pressure*OD/(2*yield_strengths.loc[k,'SMYS [Mpa]']*design_factor*joint_factor*mat_perf_factor)
        thickness.append(thickness_k)
        if max(schedules_DN['Wall thickness [mm]']) >= thickness_k:
            closest_row = schedules_DN.loc[schedules_DN['Wall thickness [mm]'] >= thickness_k]
            closest_thickness = closest_row['Wall thickness [mm]'].min()
            closest_schedule = schedules_DN.loc[schedules_DN['Wall thickness [mm]']==closest_thickness,'Schedule'].tolist()
            schedule_minviable.append(closest_schedule)
            new_pressure_k = closest_thickness/OD*(2*yield_strengths.loc[k,'SMYS [Mpa]']*design_factor*joint_factor*mat_perf_factor)
        else:
            schedule_minviable.append([np.nan])
            new_pressure_k = pressure

        new_pressure.append(new_pressure_k)

    return (thickness,schedule_minviable,new_pressure)

def ASME_hf(spec_min_yield_strength,spec_min_tens_strength):
    # Create arrays of material performance factors based on SMYS and SMTS for ASME
    # B31.12 Table IX-5A for stresses and design pressure in MPa
    if spec_min_yield_strength <= 358.528 or spec_min_tens_strength <= 455.054:
        h_f_array = np.array([1,1,0.954,0.91,0.88,0.84,0.78])
    elif spec_min_yield_strength <=413.686 and (spec_min_tens_strength > 455.054 and spec_min_tens_strength <=517.107):
        h_f_array = np.array([0.874,0.874,0.834,0.796,0.77,0.734,0.682])
    elif spec_min_yield_strength <= 482.633 and (spec_min_tens_strength > 517.107 and spec_min_tens_strength <= 565.370):
        h_f_array = np.array([0.776,0.776,0.742,0.706,0.684,0.652,0.606])
    elif spec_min_yield_strength <= 551.581 and (spec_min_tens_strength >565.370 and spec_min_tens_strength <= 620.528):
        h_f_array = np.array([0.694,0.694,0.662,0.632,0.61,0.584,0.542]) 
    return h_f_array

def compressor_locations(casestudy_path,casestudy_name):
    # Read in compressor and pipe info from network file
    network_compressors = pd.read_excel(casestudy_path + casestudy_name,sheet_name = "GCS",index_col = None,usecols = ["Name","FromName","ToName","InService  = True","Visible  = True"]).sort_values('FromName')
    network_pipes = pd.read_excel(casestudy_path + casestudy_name,sheet_name = "GPI",index_col = None, usecols = ["Name","FromName","ToName","L [km] = 10"]).rename(columns = {"L [km] = 10":"Pipe Length [km]"})
    pipe_name = network_pipes['Name']
    # Identify from nodes for active compressors
    compressor_fromnodes = pd.DataFrame(network_compressors["FromName"])

    # Identify what pipes end in compressors and sum up the lengths prior to them
    network_pipes['Ends in comp'] = False
    network_pipes.loc[network_pipes['ToName'].isin(compressor_fromnodes["FromName"]),'Ends in comp']=True
    network_pipes['Cumulative length [km]'] = network_pipes['Pipe Length [km]'].cumsum()
    
    # Create new dataframe for outputing
    compressor_specs = network_compressors.drop(["InService  = True","Visible  = True"],axis=1)

    # Down-select cumulative lengths to those associated with compressor positions
    cum_length_comps = network_pipes.loc[network_pipes['Ends in comp']==True].copy()
    cum_length_comps['Cumulative length [mi]'] = cum_length_comps['Cumulative length [km]']*0.621
    cum_length_comps = cum_length_comps.drop(['Name','Ends in comp'],axis=1)

    # Format output dataframe
    compressor_specs = compressor_specs.set_index('FromName')
    cum_length_comps = cum_length_comps.set_index('ToName')
    compressor_specs = pd.concat([compressor_specs,cum_length_comps],axis=1).reset_index().drop(['index','Pipe Length [km]'],axis=1)

    return(compressor_specs)

# Define function for identifying pipe segments in between compressor stations
def pipesegments(casestudy_path,casestudy_name):
    # Read in compressor and pipe info from network file
    network_compressors = pd.read_excel(casestudy_path + casestudy_name + "_Network.xlsx",sheet_name = "GCS",index_col = None,usecols = ["Name","FromName","ToName","InService  = True","Visible  = True"])
    network_pipes = pd.read_excel(casestudy_path + casestudy_name + "_Network.xlsx",sheet_name = "GPI",index_col = None, usecols = ["Name","FromName","ToName","D [mm] = 600"]).rename(columns = {"D [mm] = 600":"Pipe Diameter [mm]"})
    pipe_name_diameter = network_pipes[['Name','Pipe Diameter [mm]']]
    
    # Identify from and to nodes for active compressors
    compressor_fromnodes = pd.DataFrame(network_compressors.loc[network_compressors["Visible  = True"] == True,"FromName"])
    compressor_tonodes = pd.DataFrame(network_compressors.loc[network_compressors["Visible  = True"] == True,"ToName"])
    
    # Identify the ends and starts of pipe segments between compressors
    pipe_segment_ends = network_pipes.loc[network_pipes["ToName"].isin(compressor_fromnodes["FromName"])].reset_index().drop(labels = ['index'], axis = 1)
    pipe_segment_starts = network_pipes.loc[network_pipes["FromName"].isin(compressor_tonodes["ToName"])].reset_index().drop(labels = ['index'], axis = 1)
    #Add the first and last pipes to the starts and ends
    pipe_start = pd.DataFrame(network_pipes.iloc[0]).T
    pipe_end = pd.DataFrame(network_pipes.iloc[-1]).T
    pipe_segment_starts = pd.concat([pipe_start,pipe_segment_starts],ignore_index=True)
    pipe_segment_ends = pd.concat([pipe_segment_ends,pipe_end],ignore_index=True)
    
    #Create a list of lists of pipes within each individual pipe segment and return
    num_pipe_segments = pipe_segment_starts.shape[0]
    
    pipe_segments = []

    for j in range(num_pipe_segments):
        # Identify names of pipes in between each compressor and their diameters, as well as unique number of diameters
        pipe_segment_names_j = pipe_name_diameter.loc[pipe_name_diameter.index[pipe_name_diameter['Name'] == pipe_segment_starts.loc[j,'Name']].values[0]:pipe_name_diameter.index[pipe_name_diameter['Name'] == pipe_segment_ends.loc[j,'Name']].values[0],'Name']
        pipe_segment_diameters_j = pipe_name_diameter.loc[pipe_name_diameter['Name'].isin(pipe_segment_names_j),['Name','Pipe Diameter [mm]']]
        pipe_segment_diameters_unique_j = list(set(pipe_segment_diameters_j['Pipe Diameter [mm]']))
        
        # Check if there are more than one unique diameters in between compressors. If so, make each diameter a unique segment
        if len(pipe_segment_diameters_unique_j)<=1:
            pipe_segments.append(pipe_segment_names_j.tolist())
        elif len(pipe_segment_diameters_unique_j)>1:
            for k in range(len(pipe_segment_diameters_unique_j)):
                pipe_segments.append(pipe_segment_diameters_j.loc[pipe_segment_diameters_j['Pipe Diameter [mm]']==pipe_segment_diameters_unique_j[k],'Name'].tolist())
    
    return(pipe_segments)


def setNodeMaxPressure(mydll, pipe_params,numPipes):
    # Function name: setNodeMaxPressure(dllfolder, casestudy_name)
    # SAInt does not have a parameter for pipes that constrains maximum operating pressure
    # However, it does for network node. This function is written to attribute pipeline maximum 
    # allowable operating pressure (MAOP) to network nodes.
    
    # Get pipe node connections data from SAInt and place into dataframe called pipe_node_set
    # (this data could be pulled from the network file instead for BlendPATH/SAInt separation)
    pipe_name = []
    pipe_from_node = []
    pipe_to_node = []
    
    for j in range(numPipes):
        pipe_name.append(mydll.evalStr(f'GPI.{j}.Name'))
        pipe_from_node.append(mydll.evalStr(f'GPI.{j}.FromName'))
        pipe_to_node.append(mydll.evalStr(f'GPI.{j}.ToName'))
    pipe_name = pd.DataFrame(pipe_name,columns = ['Pipe Name'])
    pipe_node_set = pipe_name.join(pd.DataFrame(pipe_from_node,columns = ['From Node']))    
    pipe_node_set = pipe_node_set.join(pd.DataFrame(pipe_to_node,columns = ['To Node']))
    
    # Loop through pipe_node_set to obtain a list that attribute pipeline connections to 
    # network nodes
    numNO = mydll.evalInt("GNET.NUMNO")
    
    node_list = []
    node_list_pipes = []
    
    for i in range(numNO):
        pipe_connections = []
        node_list.append(mydll.evalStr(f'GNO.{i}.Name'))
        for j in range(numPipes):
            if pipe_node_set["From Node"].iloc[j] == mydll.evalStr(f'GNO.{i}.Name'):
                pipe_connections.append(pipe_node_set["Pipe Name"].iloc[j])        
            if pipe_node_set["To Node"].iloc[j] == mydll.evalStr(f'GNO.{i}.Name'):
                pipe_connections.append(pipe_node_set["Pipe Name"].iloc[j]) 
        node_list_pipes.append(pipe_connections)
    
    # Replace pipe name data in node_list_pipe with MAOP data from pipe_params 
    # Note: I could have copied node_list_pipe into another list called node_list_MAOP here
    # but to do this properly, I would have to import a dependency called copy. If we
    # want to create another list but without importing copy, both lists will be linked.
    for i in range(len(node_list_pipes)):
        if node_list_pipes[i] == []:
            node_list_pipes[i] = None
        else:
            for j in range(len(node_list_pipes[i])):
                pipe = node_list_pipes[i][j]
                pipe_param_index = pipe_params.loc[lambda param: param['Pipe Name'] == pipe].index.item()
                node_list_pipes[i][j] = pipe_params["ASME Design Pressure [MPa]"].iloc[pipe_param_index]
    
    # Reduce list values in node_list_pipes to scalar minimum values. This is to assign the minimum MAOP
    # of the pipelines connected to a node max pressure limit
    
    for i in range(len(node_list_pipes)):
        if node_list_pipes[i] == None:
            continue
        else:
            node_list_pipes[i] = min(node_list_pipes[i])
    
    # Take lists node_list and node_list_pipes and consolidate them into one DataFrame
    
    node_list_MAOP = pd.DataFrame(node_list, columns = ["Node Name"])
    node_list_MAOP = node_list_MAOP.join(pd.DataFrame(node_list_pipes, columns = ['Max Allowable Pressure [MPa]']))
    
    return node_list_MAOP


def segment_props(mydll,casestudy_path,casestudy_name,design_pressure_ASME,pipe_params,numPipes,comp_pressure_ratio,final_outlet_pressure):
    
    pipe_segments = pipesegments(casestudy_path,casestudy_name)

    network_pipes = pd.read_excel(casestudy_path + casestudy_name + "_Network.xlsx",sheet_name = "GPI",index_col = None, usecols = ["Name","FromName","ToName"])

    
    # Read in pipe fluid props necessary for module
    pipe_fluidprops = getpipefluidprops(mydll,numPipes)
    #   This calculation is off from what SAInt gives for the kg/s (JK Mar 28 2023). But causes an issue some additional compressor sceanrios (JK 4/7/23)
    pipe_fluidprops['Inlet Mass Flow Rate [kg/s]'] = pipe_fluidprops['Inlet Density [kg/m3]']*pipe_fluidprops['Inlet Vol Flow [m3/s]']
    pipe_fluidprops['Outlet Mass Flow Rate [kg/s]'] = pipe_fluidprops['Outlet Density [kg/m3]']*pipe_fluidprops['Outlet Vol Flow [m3/s]']
    pipe_fluidprops['Temperature [K]'] = pipe_fluidprops['Temperature [C]'] + 273.15
    
    # Read in node fluid props because gas contant is not provided for the pipe object in SAInt
    node_fluidprops = getnodefluidprops(mydll,numPipes)
    # Get gas constant into the same dataframe as the rest of the properties
    num_nodes = node_fluidprops.shape[0]
    num_pipes = pipe_fluidprops.shape[0]
    gas_constant = []
    for j in range(num_pipes):
        for k in range(num_nodes):
            if (network_pipes["FromName"].iloc[j] == node_fluidprops['Node Name'].iloc[k]):
                gas_constant_j = node_fluidprops["Gas Constant [J/kg-K]"].iloc[k]
        gas_constant.append(gas_constant_j)
    gas_constant = pd.DataFrame(gas_constant,columns = ['Gas Constant [J/kg-K]'])
    pipe_fluidprops = pipe_fluidprops.join(gas_constant)
    
    #Add in ASME design pressure
    pipe_fluidprops = pipe_fluidprops.join(design_pressure_ASME)
    
    num_pipe_segments = len(pipe_segments)
    
    # Determine inlet pressure and ASME design pressure
    segment_inlet_pressure = []
    segment_asme_design_pressure = []
    for j in range(num_pipe_segments):    
        #Limit fluid props to those in current segment
        pipes_in_segment = pipe_segments[j]
        segment_fluidprops = pipe_fluidprops.loc[pipe_fluidprops['Pipe Name'].isin(pipes_in_segment)]
    
        # Set inlet pressure of each segment to the MAOP from ASME
        segment_inlet_pressure.append(max(segment_fluidprops['ASME Design Pressure [MPa]'])*1e6)
        segment_asme_design_pressure.append(max(segment_fluidprops['ASME Design Pressure [MPa]']))
        
    
    # Establish outlet pressure. For now, this is just MAOP divide by compressor pressure ratio.       
    segment_outlet_pressure = []
    for j in range(num_pipe_segments-1):
        # Establish compressor pressure ratio
        segment_outlet_pressure.append(segment_inlet_pressure[j+1]/comp_pressure_ratio)
    
    # Set final outlet pressure. Preferably based on end-use requirement or user input
    segment_outlet_pressure.append(final_outlet_pressure)
    
    # Set the rest of the segment properties
    segment_props_names = ['Inlet pressure [Pa]','ASME design pressure [MPa]','Outlet pressure [Pa]','Inlet Mass flow rate [kg/s]',\
                           'Outlet Mass flow rate [kg/s]','Inlet density [kg/m3]','Outlet density [kg/m3]','Diameter [m]', 'NPS', 'DN',\
                            'Length [m]','Z [-]','R [J/kg-K]','Friction factor [-]','Temperature [K]','Roughness [mm]','Elevation change [m]']
    segment_props_dict = {i:[] for i in segment_props_names}
    segment_props_dict['Inlet pressure [Pa]'] = segment_inlet_pressure
    segment_props_dict['Outlet pressure [Pa]'] = segment_outlet_pressure
    segment_props_dict['ASME design pressure [MPa]'] = segment_asme_design_pressure
    for j in range(num_pipe_segments):
        #Limit fluid props to those in current segment
        pipes_in_segment = pipe_segments[j]
        segment_fluidprops = pipe_fluidprops.loc[pipe_fluidprops['Pipe Name'].isin(pipes_in_segment)].reset_index().drop(['index'],axis = 1)
        pipe_fluidprops_j = pipe_fluidprops.loc[pipe_fluidprops['Pipe Name'].isin(pipes_in_segment)]
        # Mass Flow rate
        segment_props_dict['Inlet Mass flow rate [kg/s]'].append(segment_fluidprops.loc[0,'Inlet Mass Flow Rate [kg/s]'])
        segment_props_dict['Outlet Mass flow rate [kg/s]'].append(segment_fluidprops.loc[len(pipes_in_segment)-1,'Outlet Mass Flow Rate [kg/s]'])
        # Density
        segment_props_dict['Inlet density [kg/m3]'].append(segment_fluidprops.loc[0,'Inlet Density [kg/m3]'])
        segment_props_dict['Outlet density [kg/m3]'].append(segment_fluidprops.loc[len(pipes_in_segment)-1,'Outlet Density [kg/m3]'])
        # Diameter
        segment_props_dict['Diameter [m]'].append(max(pipe_params.loc[pipe_params['Pipe Name'].isin(pipes_in_segment),'Pipe Diameter [mm]'])*0.001)
        segment_props_dict['NPS'].append(max(pipe_params.loc[pipe_params['Pipe Name'].isin(pipes_in_segment),'NPS']))
        segment_props_dict['DN'].append(max(pipe_params.loc[pipe_params['Pipe Name'].isin(pipes_in_segment),'DN']))
        # Length
        segment_props_dict['Length [m]'].append(sum(pipe_params.loc[pipe_params['Pipe Name'].isin(pipes_in_segment),'Length [km]'])*1000)
        # Compressibility factor
        segment_props_dict['Z [-]'].append(np.mean(pipe_fluidprops_j['Compressibility Factor [-]']))
        # Gas constant
        segment_props_dict['R [J/kg-K]'].append(np.mean(pipe_fluidprops_j['Gas Constant [J/kg-K]']))
        # Friction factor
        segment_props_dict['Friction factor [-]'].append(np.mean(pipe_fluidprops_j['Friction Factor [-]']))
        #Temperature
        segment_props_dict['Temperature [K]'].append(np.mean(pipe_fluidprops_j['Temperature [K]']))
        #Roughness
        segment_props_dict['Roughness [mm]'].append(np.mean(pipe_fluidprops_j['Roughness [mm]']))
        #Elevation change
        segment_props_dict['Elevation change [m]'].append(sum(pipe_fluidprops_j['Elevation change [m]']))

    segment_props = pd.DataFrame(segment_props_dict)
    
    return(segment_props)


def get_mechanical_properties(path:str,pipe_params):
    # Read in mechanical properties and merge them into pipe_params
    mechanical_props = pd.read_csv(path+'_pipe_mech_props.csv',index_col = None,header = 0)

    pipe_params = pipe_params.join(mechanical_props.set_index('Pipe Name'),on = 'Pipe Name')
    # Calculate pipe pressure in MPa
    pipe_params['Max pressure [MPa]'] = (pipe_params['Max pressure [bar-g]'])/10

    

    return pipe_params

def get_design_pressure_ASME(numPipes:int,design_option:str,location_class:int,joint_factor:int,pipe_params):
    '''
        Determined the design pressure as a dataframe

        Parameters:
        -------------
        numPipes: int
            Number of pipes
        design_option: str
            design option choice: ex. "No fracture control"
        location_class: int
            -
        joint_factor: int
            -
        pipe_params: Dataframe
            pd Dataframe with columns of "Yield strength [Mpa]", "Tensile strength [Mpa]", "Max pressure [MPa]", "Pipe Diameter [mm]","Wall thickness [mm]"
    '''
    
    design_pressure_ASME = []
    for j in range(numPipes):
        design_pressure_ASME.append(\
            designpressure_ASMEB3112(design_option,location_class,joint_factor,\
                                            pipe_params.loc[j,'Yield strength [Mpa]'],\
                                            pipe_params.loc[j,'Tensile strength [Mpa]'],\
                                            pipe_params.loc[j,'Max pressure [MPa]'],\
                                            pipe_params.loc[j,'DN']/1000,\
                                            pipe_params.loc[j,'Wall thickness [mm]']/1000\
                                        )\
                                    )

    # Convert ASME design pressure to data frame
    design_pressure_ASME = pd.DataFrame(design_pressure_ASME,columns = ['ASME Design Pressure [MPa]'])
    return design_pressure_ASME

def get_design_pressure_violations(pipe_params,design_pressure_ASME,pipe_segments,params,segment_props):

    # Initialize design_pressure_violation dataframe

    pipe_name = pipe_params['Pipe Name'].to_frame()
    design_pressure_violations = pipe_name.join(design_pressure_ASME)
    
    # Add segment_name as a column to design_pressure_violation
    segment_name = []
    segment_index = 0
    for segment in pipe_segments: 
        for pipe in segment:
            segment_name.append(segment_index)
        segment_index += 1
    
    design_pressure_violations = design_pressure_violations.join(pd.DataFrame(segment_name, columns = ["Segment Name"]))

    # Identify rows (or pipes) with exceeded MAOP
    design_pressure_violations['Design Pressure Difference [MPa]'] = design_pressure_ASME['ASME Design Pressure [MPa]'] - pipe_params['Max pressure [MPa]']

    design_pressure_violations.loc[(design_pressure_violations["Design Pressure Difference [MPa]"] >=0),'Design Pressure Violation'] = 0
    design_pressure_violations.loc[(design_pressure_violations["Design Pressure Difference [MPa]"] <0),'Design Pressure Violation'] = 1

    # Check custom hydraulic model to see if it also shows design pressure violations
    pressure_drop_factor = params['pressure_drop_factor']
    nodes = params['nodes']
    T_c = params['T_c']
    P_c = params['P_c']
    viscosity = params['viscosity']

    #   Set low limit on segment outlet pressures. Current limit is set to 20 bar which is around the inlet pressure of a city gates.
    #   Adjusting to a lower low limit may cause instability with fsolve as pressure difference (P2^2 - P1^2) becomes highly non-linear at low pressures
    p_out_lim = pd.DataFrame(2000000, index=np.arange(len(segment_props)), columns=["Low Pressure Limit [Pa]"])
    p_out_lim["Low Pressure Limit [Pa]"].iloc[-1] = segment_props["Outlet pressure [Pa]"].iloc[-1]
    segment_inlet_pressures = []

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
        inlet_pressure_segment_j = fsolve(mod_fxn.singlepipe_getinletpressure_knownlength_discretized,x_init,args = args)

        segment_inlet_pressures.append(inlet_pressure_segment_j[0])

    # Initialize excess pressure drop violation dataframe
    pressure_drop_violations_custom_hydraulics = pd.DataFrame(0, index=np.arange(len(segment_props)), columns=["Inlet pressure violation"])
    pressure_drop_violations_custom_hydraulics = pressure_drop_violations_custom_hydraulics.join(pd.DataFrame(segment_inlet_pressures, columns=["Inlet pressure [Pa]"]))
    pressure_drop_violations_custom_hydraulics = pressure_drop_violations_custom_hydraulics.join(segment_props["ASME design pressure [MPa]"]*1000000)
    pressure_drop_violations_custom_hydraulics = pressure_drop_violations_custom_hydraulics.rename(columns = {"ASME design pressure [MPa]": 'ASME design pressure [Pa]'})

    # Identify segments that have pressures that drop below ambient or terminal demand node low pressure limit
    pressure_drop_violations_custom_hydraulics['Inlet Pressure Difference [Pa]'] = pressure_drop_violations_custom_hydraulics["ASME design pressure [Pa]"] - pressure_drop_violations_custom_hydraulics["Inlet pressure [Pa]"] 
    pressure_drop_violations_custom_hydraulics.loc[(pressure_drop_violations_custom_hydraulics['Inlet Pressure Difference [Pa]'] >=0),'Design pressure violation'] = 0
    pressure_drop_violations_custom_hydraulics.loc[(pressure_drop_violations_custom_hydraulics['Inlet Pressure Difference [Pa]'] <0),'Design pressure violation'] = 1

    for ind,segment in enumerate(pipe_segments):
        design_pressure_violations.loc[design_pressure_violations['Pipe Name'].isin(segment),'Design Pressure Violation']=pressure_drop_violations_custom_hydraulics.loc[ind,'Design pressure violation']

    design_pressure_violations['Design Pressure Violation'] = design_pressure_violations['Design Pressure Violation'].astype(int)


    return design_pressure_violations

def get_yields_and_schedules():
    yield_strengths = pd.read_csv(resource_filename(__name__,'resources/steel_mechanical_props.csv'),index_col = None,header = 0)
    # Drop yield strengths for which we have no costs. Eventually need to make fix for this
    yield_strengths = yield_strengths.drop([0,1,2,9],axis = 0).reset_index()
    # Read in CSV with pipe schedules
    schedules = pd.read_csv(resource_filename(__name__,'resources/pipe_dimensions_metric.csv'),index_col = None,header = 0)

    return yield_strengths,schedules

def get_design_pressure(design_pressure_known:bool,pipe_segments:int,path:str,pipe_params):

    # Read in design pressure if known (True). If not known, estimate design pressure for each pipe segment as the max for those pipes
    if design_pressure_known == True:
        design_pressure = pd.read_csv(path+'_design_pressure.csv',index_col = None,header = 0)
        design_pressure = design_pressure['Max pressure [Mpa]']
    else:
        num_pipe_segments = len(pipe_segments)
        segment_maxpressure = []
        design_pressure = []

        for j in range(num_pipe_segments):    
            segment_maxpressure.append(max(pipe_params.loc[pipe_params["Pipe Name"].isin(pipe_segments[j]),"Max pressure [MPa]"]))
            num_pipes_persegment = len(pipe_segments[j])

            design_pressure.extend([segment_maxpressure[j]]*num_pipes_persegment)

        design_pressure = pd.Series(design_pressure,name = 'Max pressure [MPa]')

    return design_pressure

def get_demand_nodes(pipe_segments:object,path:str):
    network_pipes = pd.read_excel(f'{path}_Network.xlsx',sheet_name = "GPI",index_col = None, usecols = ["Name","FromName","ToName"])
    demand_nodes = pd.read_excel(f'{path}_Network.xlsx',sheet_name = "GDEM",index_col = None,usecols = ["Name","NodeName"])     

    num_segments = len(pipe_segments)

    segment_demand_nodes = []

    for j in range(num_segments): 
        pipes_in_segment = network_pipes.loc[network_pipes["Name"].isin(pipe_segments[j])]
        to_nodes_in_segment = pd.melt(pipes_in_segment,id_vars = None,value_vars = ['ToName']).rename(columns = {'value':'Name'}).drop(['variable'],axis = 1)
        to_nodes_in_segment = pd.DataFrame(to_nodes_in_segment.Name.unique(), columns = ['Name'])
        demand_nodes_in_segment = to_nodes_in_segment.loc[to_nodes_in_segment['Name'].isin(demand_nodes["NodeName"])].reset_index().drop(['index'], axis=1)
        demand_nodes_in_segment = demand_nodes_in_segment['Name'].tolist()
        segment_demand_nodes.append(demand_nodes_in_segment)

    return network_pipes,segment_demand_nodes

def get_compressor_usage(mydll,CS_name,units = 'MMBTU/h'):
   
    CS_PR = []
    CS_fuel = []
    CS_rating = []

    for j in CS_name:
        CS_PR.append(mydll.evalFloat(f'GCS.{j}.PR'))
        CS_fuel.append(mydll.evalFloat(f'GCS.{j}.FUEL.[{units}]')) # Loop through CS_name values to pull compressor fuel consumption output in MMBTU/hr units into a list parameter
        CS_rating.append(mydll.evalFloat(f'GCS.{j}.POWS.[MW]')) # Same as the above line but with compressor station power operating point in units of kW

    comp_name = pd.DataFrame(CS_name, columns = ['Name'])
    comp_PR = pd.DataFrame(CS_PR,columns = ['Pressure ratio [-]'])
    comp_fuel = pd.DataFrame(CS_fuel,columns=[f'Fuel Consumption [{units}]'])
    comp_rating = pd.DataFrame(CS_rating,columns=['Compressor Rating [MW]'])

    comp_params = pd.concat([comp_name,comp_PR,comp_fuel,comp_rating],axis=1)

    return CS_fuel,CS_rating,comp_params


def get_demand_constraints(casestudy_path,casestudy_name):
    demand_node_names = pd.read_excel(casestudy_path + casestudy_name + '_Network_new.xlsx',sheet_name = "GDEM",index_col = None,usecols = ["Name"])
    
    demand_node_names_list = demand_node_names.values.tolist()

    constraint_names = []
    for j in range(len(demand_node_names)):
        constraint_names.append('GDEM.'+demand_node_names_list[j][0]+'.QSET')

    boundary_conditions = pd.read_excel(casestudy_path + casestudy_name + '_Event_new.xlsx',sheet_name = 'GSCE',index_col=None,usecols=['Parameter','Value','Unit'])

    demand_constraints = boundary_conditions.loc[boundary_conditions['Parameter'].isin(constraint_names)]

    demand_constraints = demand_constraints.reset_index(drop=True)

    demand_constraints = pd.concat([demand_node_names,demand_constraints],axis=1).drop(labels = ['Parameter','Unit'], axis = 1)

    return(demand_constraints)

def get_demand_flows(mydll,casestudy_path,casestudy_name):

    demand_node_names = pd.read_excel(casestudy_path + casestudy_name + '_Network_new.xlsx',sheet_name = "GDEM",index_col = None,usecols = ["Name"])

    demand_flow = []

    for j in demand_node_names['Name'].values:
        demand_flow.append(mydll.evalFloat(f'GDEM.{j}.Q[MW]'))

    demand_flow = pd.DataFrame(demand_flow,columns = ['Value'])

    demand_flow = pd.concat([demand_node_names,demand_flow],axis=1)

    return(demand_flow)


def get_fuel_flow(p_1,p_2,X,T_K,fuel_GCV,m_dot):

    # Compressor isentropic efficiency
    eta_comp_s = 0.78

    # State 1
    state_1 = ct.Solution(resource_filename(__name__,'resources/gri30_rk.yaml'), transport=None)
    state_1.TPX = T_K,p_1,X
    h_1 = state_1.h
    s_1 = state_1.s

    # State 2 isentropic (at same entropy, different pressure)
    s_2_s = s_1
    state_2_s = ct.Solution(resource_filename(__name__,'resources/gri30_rk.yaml'), transport=None)
    state_2_s.SPX = s_2_s,p_2,X
    h_2_s = state_2_s.h

    # Enthalpy change (isentropic)
    delta_h_s = h_2_s - h_1
    # Enthalpy real
    delta_h = delta_h_s/eta_comp_s

    # State 2 real
    # Enthalpy for state 2
    h_2 = h_1 + delta_h
    state_2 = ct.Solution(resource_filename(__name__,'resources/gri30_rk.yaml'), transport=None)
    state_2.HPX = h_2,p_2,X
    # temp_2 = state_2.T

    # Calculate shaft power

    W_dot_shaft = m_dot*delta_h/1000

    eta_mech = 0.357

    driver_fuel_rate_kW = W_dot_shaft/eta_mech


    state_fuel = ct.Solution(resource_filename(__name__,'resources/gri30_rk.yaml'), transport=None)
    state_fuel.TPX = T_K,101325,X
    rho_fuel = state_fuel.density

    fuel_flow_kgps = driver_fuel_rate_kW/1000/fuel_GCV*rho_fuel

    fuel_flow_sm3ps = driver_fuel_rate_kW/1000/fuel_GCV

    return fuel_flow_kgps