#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:19:54 2022

@author: raul
"""

import numpy as np
from scipy.optimize import fsolve

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
def resistance_coefficient(diameter, up_dn, down_dn) -> float:
  '''Return the Resistance coefficients of all fittings attached to 
  the control valve'''
  diameter *= 1000; up_dn *= 1000; down_dn *= 1000
  reducer  =  0.5 * ((1-(diameter/up_dn)**2)**2)
  diffuser =  ((1-(diameter/down_dn)**2)**2)
  bernulli =  (diameter/down_dn)**4 - (diameter/up_dn)**4
  return reducer + diffuser + bernulli

def piping_geometry_factor(f_coefficent, diameter, up_dn, down_dn) -> float:
  '''Return the piping geometry factor Fp'''
  diameter *= 1000; up_dn *= 1000; down_dn *= 1000
  rc = resistance_coefficient(diameter/1000, up_dn/1000, down_dn/1000)
  return (1 / np.sqrt(1+(rc*(f_coefficent/diameter**2)**2)/0.0016))
v_piping_geometry_factor = np.vectorize(piping_geometry_factor)


def combined_geometry_factor(f_coefficent, fl, diameter, up_dn, down_dn) -> float:
  '''
  Return the Combined liquid pressure recovery factor flp
  '''
  rc = resistance_coefficient(diameter/1000, up_dn/1000, down_dn/1000)
  return (fl / np.sqrt(1+(rc*(f_coefficent/diameter**2)**2)*(fl**2)/0.0016))
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
v_permissible_differential_pressure = np.vectorize(max_differential_pressure)


#-------------------------------------------------------------------------------

# Plot functions for the Kv/Kvs
def drm_ll3(openinig,b,d,e):
  return d/(1+np.exp(b*(np.log(openinig)-np.log(e))))

# Plot functions for the Liquid pressure recovery factor Fl
def pressure_recovery_factor(openinig, fls, b, d, e) -> float:
  '''
  fl The liquid pressure recovery factor, fl, predicts the amount of pressure 
  '''
  sigma_value = 1/(fls**2) - 1
  kv_kvs = drm_ll3(openinig, b, d, e)
  return np.sqrt(1/(sigma_value * kv_kvs + 1))
v_pressure_recovery_factor = np.vectorize(pressure_recovery_factor)

# Incipient Cavitation
def incipient_cavitation(openinig, fls, b, d, e) -> float:
  xfz = 0.71
  kv_kvs = drm_ll3(openinig, b, d, e)
  return (1/(xfz * fls**2) - 1) * kv_kvs

# Solve kv_kvs function
def root_drm_ll3(kv_kvs,b,d,e):
  def fun(x,kv_kvs,b,d,e):
    return d/(1+np.exp(b*(np.log(x)-np.log(e))))-kv_kvs
  root = fsolve(fun, 50, args=(kv_kvs,b,d,e))
  return root
v_root_drm_ll3 = np.vectorize(root_drm_ll3)

#-------------------------------------------------------------------------------

