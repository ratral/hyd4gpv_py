"""
Created on Sun Sep 25 16:19:54 2022
@author: Dr.Trujillo
"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from numpy.core.fromnumeric import mean

#-------------------------------------------------------------------------------
def atm_pressure(masl) -> float:
  """Returns: Atmosphere  Pressure in [bar]"""
  return (1/1000)*((44331.514 - masl)/11880.516)**(1/0.1902632)
v_atm_pressure = np.vectorize(atm_pressure)


def vapor_pressure(temp_c) -> float:
  """Returns: Vapor pressure [bar]"""
  return (0.61121*np.exp((18.678-temp_c/234.5)*(temp_c/(257.14+temp_c))))/100
v_vapor_pressure = np.vectorize(vapor_pressure)


def density(temp_c) -> float:
  """Returns: density of water in [kg/m³]"""
  temp_k = temp_c + 273.15
  a = 0.14395; b = 0.0112; c = 649.727; d = 0.05107
  return  a/(b**(1 + (1-temp_k/c)**d))
v_density = np.vectorize(density)


def velocity(flow, diameter) -> float:
  """Returns: Velocity in [m/s]"""
  return (flow/3600)/((np.pi*diameter**2)/4)
v_velocity = np.vectorize(velocity)


def velocity_factor(flow, diameter) -> float:
  """Returns: Velocity factor in [bar]"""
  return (v_velocity(flow, diameter)**2 / (2*9.80665)) / 10
v_velocity_factor = np.vectorize(velocity_factor)


def absolute_pressure(gauge_pressure, masl) -> float:
  """Returns: Absolute Pressure in [bar]"""
  ap = v_atm_pressure(masl)
  return gauge_pressure + ap
v_absolute_pressure = np.vectorize(absolute_pressure)


def sigma_0(p_up, p_down, masl, temp_c) -> float:
  p1 = v_absolute_pressure(p_up, masl)
  p2 = v_absolute_pressure(p_down, masl)
  return (p1 - v_vapor_pressure(temp_c))/(p1-p2)
v_sigma_0 = np.vectorize(sigma_0)


def sigma_1(p_up, p_down, masl, temp_c) -> float:
  p1 = v_absolute_pressure(p_up, masl)
  p2 = v_absolute_pressure(p_down, masl)
  return (p2 - v_vapor_pressure(temp_c))/(p1-p2)
v_sigma_1 = np.vectorize(sigma_1)


def sigma_2(p_up, p_down, flow, diameter, masl, temp_c) -> float:
  p1 = v_absolute_pressure(p_up, masl)
  p2 = v_absolute_pressure(p_down, masl)
  v_factor = velocity_factor(flow, diameter)
  return (p2 - v_vapor_pressure(temp_c))/(p1 - p2 + v_factor)
v_sigma_2 = np.vectorize(sigma_2)


def flow_coefficent(p_up, p_down, flow, temp_c) -> float:
  ''' Return Flow Coefficent Kv in m3/h'''
  return flow*np.sqrt((v_density(temp_c)/1000)/(p_up - p_down))
v_flow_coefficent = np.vectorize(flow_coefficent)


def drop_coefficient(p_up, p_down, flow, diameter, temp_c) -> float:
  ''' Return Pressure Drop Coefficient - zeta'''
  kv = v_flow_coefficent(p_up, p_down, flow, temp_c)
  return (1/626.3)*((diameter*1000)**2/kv)**2
v_drop_coefficient = np.vectorize(drop_coefficient)


def kv_fun_zeta(diameter, zeta_value) -> float:
  ''' Return Flow Coefficent Kv in m3/h'''
  return ((diameter*1000)**2)/np.sqrt(626.3*zeta_value)
v_kv_fun_zeta = np.vectorize(kv_fun_zeta)

#-------------------------------------------------------------------------------
def resistance_coefficient(diameter, dn_up, dn_down, z_plate = 0) -> float:
  '''
  Return the Resistance coefficients of all fittings 
  attached to the control valve
  '''
  reducer  =  0.5 * ((1-(diameter/dn_up)**2)**2)
  diffuser =  ((1-(diameter/dn_down)**2)**2)
  bernulli =  (diameter/dn_down)**4 - (diameter/dn_up)**4
  return reducer + diffuser + bernulli + z_plate


def piping_geometry_factor(f_coefficent, diameter, dn_up, dn_down, z_plate) -> float:
  '''Return the piping geometry factor fp'''
  rc = resistance_coefficient(diameter, dn_up, dn_down, z_plate)
  return (1 / np.sqrt(1+(rc*(f_coefficent/(diameter*1000)**2)**2)/0.0016))
v_piping_geometry_factor = np.vectorize(piping_geometry_factor)


def combined_geometry_factor(f_coefficent, fl, diameter, dn_up, dn_down, z_plate) -> float:
  ''' Return the Combined liquid pressure recovery factor flp'''
  rc = resistance_coefficient(diameter, dn_up, dn_down, z_plate)
  return (fl / np.sqrt(1+(rc*(f_coefficent/(diameter*1000)**2)**2)*(fl**2)/0.0016))
v_combined_geometry_factor = np.vectorize(combined_geometry_factor)


def critical_pressure_factor(temp_c) -> float:
  ''' ff is the Liquid critical pressure ratio factor'''
  pv = v_vapor_pressure(temp_c)
  #  the critical thermodynamic pressure for water is 221.2 bar
  pc = 221.2
  return 0.96-0.28*np.sqrt(pv/pc)
v_critical_pressure_factor = np.vectorize(critical_pressure_factor)


def max_differential_pressure(flp, fp, p1, temp_c) -> float:
  '''The maximum permissible differential pressure'''
  pv = v_vapor_pressure(temp_c)
  ff = v_critical_pressure_factor(temp_c)
  return ((flp/fp)**2)*(p1-ff*pv) 
v_max_differential_pressure = np.vectorize(max_differential_pressure)

#-------------------------------------------------------------------------------

def drm_ll3(openinig,b,d,e) -> float:
  '''# Plot functions for the Kv/Kvs'''
  return d/(1+np.exp(b*(np.log(openinig)-np.log(e))))


def root_drm_ll3(kv_kvs,b,d,e) -> float:
  '''Solve kv_kvs function'''
  def fun(x,kv_kvs,b,d,e):
    return d/(1+np.exp(b*(np.log(x)-np.log(e))))-kv_kvs
  root = fsolve(fun, 50, args=(kv_kvs,b,d,e))
  return root
v_root_drm_ll3 = np.vectorize(root_drm_ll3)

#-------------------------------------------------------------------------------

def pressure_recovery_factor(openinig, fls, b, d, e) -> float:
  ''' functions for the Liquid pressure recovery factor Fl'''
  sigma_value = 1/(fls**2) - 1
  kv_kvs = drm_ll3(openinig, b, d, e)
  return np.sqrt(1/(sigma_value * kv_kvs + 1))
v_pressure_recovery_factor = np.vectorize(pressure_recovery_factor)


def cavitation(cav_type, openinig, flps, b, d, e) -> float:
  '''Cavitation Curves'''
  factor_cav = {'incipient':0.71, 'constant':0.81, 'maximum':1}
  k = factor_cav[cav_type]
  kv_kvs = drm_ll3(openinig, b, d, e)
  return (1/(k * flps**2) - 1) * kv_kvs


def cavitation_regim(s , sigma_i, sigma_c, sigma_m)-> int:
  if s <= sigma_m:
    return 3
  elif (s > sigma_m) and  (s <= sigma_c):
    return 2
  elif (s > sigma_c) and  (s <= sigma_i):
    return 1
  else:
    return 0
v_cavitation_regim = np.vectorize(cavitation_regim)

#-------------------------------------------------------------------------------
def convert_to_dataf(np_data):
    '''Convert the projoject imput data into a Data Frame'''
    dataf = (
        pd.DataFrame(np_data, columns = ['condition', 'p_up', 'p_down', 'flow'])
        .astype({'condition':'U16', 
                 'p_up':'float64',
                 'p_down':'float64', 
                 'flow':'float64'})
    )    
    return dataf


def calculation_operating_data(dataf, masl, diameter, temp_c):
    '''Calculate the different values''' 
    dataf = (
      dataf
        .assign(
          p1       = v_absolute_pressure(dataf.p_up, masl),
          p2       = v_absolute_pressure(dataf.p_down, masl),
          dp       = dataf.p_up-dataf.p_down,
          velocity = v_velocity(dataf.flow, diameter),
          v_factor = velocity_factor(dataf.flow, diameter),
          sigma_0  = v_sigma_0(dataf.p_up, dataf.p_down, masl, temp_c),
          sigma_1  = v_sigma_1(dataf.p_up, dataf.p_down, masl, temp_c),
          sigma_2  = sigma_2(dataf.p_up, dataf.p_down, 
                             dataf.flow, diameter, masl, temp_c),
          kv       = v_flow_coefficent(dataf.p_up, dataf.p_down,
                                       dataf.flow, temp_c),
          zeta     = v_drop_coefficient(dataf.p_up, dataf.p_down, 
                                        dataf.flow, diameter, temp_c)
        )
    )
    
    return dataf

#-------------------------------------------------------------------------------

def load_valves_parameter(path_valves_data):
    return pd.read_csv(path_valves_data) 
  

def select_possible_values(dataf, brand, diameter):
    '''filter brand,  valves diameter and select specific columns'''
    dn = diameter*1000
    columns = ['cyl_name','kv_b','kv_d','kv_e','zvs','fls']
    dataf  = (
        dataf
        .query('brand == @brand')
        .query('dn_min <= @dn <= dn_max')
        [columns]
    )
    
    return dataf


def calc_zvs_kvs_flps(dataf, diameter, dn_up, dn_down, z_plate): 
    ''' 
    1. Calculation of the resistance coefficient of the piping
    2. Calculation of the Resistance coefficient of the valve (Zeta Value) 
       PLUS piping coefficient and zeta value of the O_plate
    '''
    dataf = (
        dataf
        .assign(
            r_coeff = resistance_coefficient(diameter, dn_up, dn_down, z_plate),
            zvs2 = dataf.zvs 
                    + resistance_coefficient(diameter, dn_up, dn_down, z_plate) 
        )
    )
    
    # Calculation of the Kvs with the Zvs2
    dataf = (
        dataf
        .assign(
            kvs = v_kv_fun_zeta(diameter, dataf.zvs2)
        )
    )
    
    # Combined liquid pressure recovery factor for full open (Flps)
    dataf = (
        dataf
        .assign(
            flps = v_combined_geometry_factor(
                dataf.kvs, dataf.fls,  diameter, 
                dn_up, dn_down, z_plate
            )
        )
    )
    return dataf

#-------------------------------------------------------------------------------

# Create combination of two pandas dataframes in two dimensions
# https://stackoverflow.com/questions/43259660/create-combination-of-two-pandas-dataframes-in-two-dimensions
# https://bitcoden.com/answers/create-combination-of-two-pandas-dataframes-in-two-dimensions


def combination_two_dataframes(dataf1, dataf2):
  dataf1['key'] = 1
  dataf2['key'] = 1
  dataf = pd.merge(dataf1, dataf2, on ='key').drop(columns=['key'])
  return dataf
  

def cal_kvkvs_sigma_regime(dataf):
  
  dataf = (
    dataf
    .assign(kv_kvs = dataf.kv/dataf.kvs)
  )

  dataf = (
      dataf
      .assign(
          position = v_root_drm_ll3(
            dataf.kv_kvs, dataf.kv_b,
            dataf.kv_d, dataf.kv_e
          )
      )
  )

  dataf = (
      dataf
      .assign(
          sigma_i = cavitation(
            'incipient', dataf.position, 
            dataf.flps, dataf.kv_b,
            dataf.kv_d, dataf.kv_e
          ),
          sigma_c = cavitation(
            'constant', dataf.position, 
            dataf.flps, dataf.kv_b,
            dataf.kv_d, dataf.kv_e
          ),
          sigma_m = cavitation(
            'maximum', dataf.position, 
            dataf.flps, dataf.kv_b,
            dataf.kv_d, dataf.kv_e
          )
        )
  )

  dataf = (
      dataf
      .assign(
          regime = v_cavitation_regim(dataf.sigma_2,
            dataf.sigma_i,
            dataf.sigma_c, 
            dataf.sigma_m)
      )
  )
  
  return dataf

def splitng_operation_data(dataf):
  
  dataf1 =  dataf[[ 
    'cyl_name','condition', 'kv_b', 'kv_d', 'kv_e', 
    'zvs', 'fls', 'r_coeff', 'kvs', 'flps'
  ]]
  
  dataf2 = dataf[[ 
    'cyl_name','condition', 'p1', 'p2', 'dp', 'flow', 'velocity',
    'kv', 'kv_kvs', 'zeta', 'sigma_2', 'position', 
    'sigma_i', 'sigma_c', 'sigma_m', 'regime'
  ]]
  
  dataf1 = (
    dataf1
    .set_index(["cyl_name"])
    .sort_index(level=["cyl_name"])
  )
  
  dataf2 = (
    dataf2
    .set_index(["cyl_name"])
    .sort_index(level=["cyl_name"])
  )
  
  return (dataf1, dataf2)


def cylinder_prioritization(dataf):
  
  dataf = (
    dataf
    .groupby("cyl_name")[["regime","position", "kv_kvs"]]
    .agg([min,max, mean])
  )
  
  # Flatten MultiIndex Columns into a Single Index
  # https://www.pauldesalvo.com/how-to-flatten-multiindex-columns-into-a-single-index-dataframe-in-pandas/
  dataf.columns = ['_'.join(col) for col in dataf.columns.values]
  
  dataf = (
    dataf
    .assign(
        kv_kvs_min = dataf.kv_kvs_min*100,
        kv_kvs_max = dataf.kv_kvs_max*100,
        position_range = dataf.position_max - dataf.position_min,
        kv_kvs_range = (dataf.kv_kvs_max - dataf.kv_kvs_min)*100
    )
  )
  
  dataf = (
    dataf
    .assign(
      valve_range = np.absolute(dataf.kv_kvs_range / dataf.position_range-1)
    )
  )
  
  dataf = dataf[["regime_min", "regime_max", "regime_mean",
                 "position_min", "position_max", "kv_kvs_min", "kv_kvs_max", 
                 "position_range", "kv_kvs_range", "valve_range"]]
                 
  dataf = dataf.sort_values(by=["regime_mean", "valve_range"],
                      ascending = [True, True])
                      
  return dataf

