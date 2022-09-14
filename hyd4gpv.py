
import math

# Water Class Properties
class Water_Properties:
    def __init__(self, tempC : float = 20)-> None:
      '''
      Parameters: 
        temp: Water temperature [Celsius]
      '''
      self.tempC = tempC
      self.tempK = tempC + 273.15

    def p_vapour(self) -> float:
      '''
      Description: Function that calculates the vapor pressure. 
      Returns: Vapor pressure [bar]
      '''
      pv = (0.61121*math.exp((18.678-self.tempC/234.5)*
                           (self.tempC/(257.14+self.tempC))))/100
      return pv

    def density(self) -> float:
      '''
      Description: Saturated water Density
      Returns: density of water in [kg/m³]
      '''
      a = 0.14395; b = 0.0112; c = 649.727; d = 0.05107
      dens = a/(b**(1+(1-self.tempK/c)**d))
      return dens

    def relative_density(self) ->float:
      '''
      Description: the ratio of the density (mass of a unit volume) of a 
      substance to the density of a given reference material. Specific gravity 
      for liquids is nearly always measured with respect to water at 
      its densest (at 3.98 °C or 39.2 °F);
      '''
      rd = self.density()/1000
      return rd

    def viscosity(self) -> float:
      '''
      Description: Dynamic Viscosity of Water
      Returns: Dynamic Viscosity of water in [mPa*s]     
      '''
      a = 1.856e-11; b = 4209; c = 0.04527; d = -3.376e-05
      visc = a * math.exp(b/self.tempK+c*self.tempK+d*self.tempK**2) 
      return visc

    def k_viscosity(self) -> float:
      '''
      Description: Kinematic Viscosity
      Returns: Kinematic Viscosity in (m2/s)*1e-6
      '''
      k_viscosity = self.viscosity()/(self.density()/1000)
      return k_viscosity

#-------------------------------------------------------------------------------
# Pipeline Properties Class with water_properties as inherent class 
class Pipe_Properties(Water_Properties):

  def __init__(self, flow: float, dn: float, tempC:float = 20)-> None:
    '''
    Parameters: 
      dn: diameter in meter (m)
      flow: flow in cubic meter per second (m³/s)
      temp: Water temperature [Celsius]
      gravity: standard acceleration due to gravity (m/s2)
    '''
    super().__init__(tempC)
    self.dn = dn
    self.flow = flow 
    self.gravity = 9.80665
  
  def velocity(self) -> float:
    '''
    Description: This function calculates the velocity of the fluid in a 
      circular pipe.
    Returns: velocity in meter per second (m/s)
    '''
    v = self.flow/((math.pi*self.dn**2)/4)
    return v
  
  def reynolds(self) -> float:
    '''
    Description: he Reynolds number (Re) is an important dimensionless quantity
     in fluid mechanics used to help predict flow patterns in different fluid
     flow situations.
    Returns: reynolds number dimensionless quantity
    '''    
    re = (4*self.flow)/(math.pi*self.dn*self.k_viscosity()*1e-6)
    return re
  
  def friction_factor(self, roughness: float) -> float:
    '''
    Description: The Colebrook–White equation, sometimes referred to simply as 
      the Colebrook equation is a relationship between the friction factor and
      the Reynolds number, pipe roughness, and inside diameter of pipe.
    Parameters: roughness in (m)
    Returns: friction factor of the pipe
    '''    
    r_roug = roughness/self.dn
    re = self.reynolds()
    f = (-2*math.log10((r_roug/3.7)-(5.02/re)*
                     math.log10(r_roug-(5.02/re)*
                              math.log10(r_roug/3.7+13/re))))**(-2)
    return f

  def head_losses(self, roughness: float, plength: float) -> float:
    '''
    Description: the Darcy–Weisbach equation is an empirical equation that 
      relates the head loss, or pressure loss, due to friction along a given 
      length of pipe to the average velocity of the fluid flow for an 
      incompressible fluid. 
    Parameters: 
      roughness: internal roughness of the pipe in meter
      plength: length of pipe in meter
    Returns:
      dw: head losses in meter (m)
    '''    
    f = self.friction_factor(roughness)
    dw = f*(plength/self.dn)*((self.velocity()**2)/(2*self.gravity))
    return dw
  
  def roughness(self, dp: float, plength: float) -> float:
    '''
    Description: Return the absolute roughness of a pipe by calibration.
      To carry out the calibration process, measuring different flow conditions
      in the pipe and the piezometric pressure drop along a previously 
      established length is necessary.
    Parameters: 
      dp: Pressure difference betwen the measured points (meter)
      plength: Length of the pipe
    Returns:
      Roughness: Absolute Roughness of the pipe in meter
    '''    
    re = self.reynolds()
    v  = self.velocity()
    f_factor = dp*(self.dn/plength)*(2*self.gravity)/(v**2)
    roug = 3.7*self.dn*(10**(-1/(2*math.sqrt(f_factor)))-2.51/(re*math.sqrt(f_factor)))
    if roug < 0:
      roug = 0
      return roug
    return roug
    


#-----------------------------------------------------------------------------------
# Class for Control Valves Properties
class control_valve_Properties(Water_Properties):
  def __init__(self, dn, flow, pu, pd, masl:float=0, tempC:float=20)-> None:
    '''
    Parameters: 
      dn: diameter in meter (m)
      flow: flow in cubic meter per second (m³/s)
      pu: Upstream Pressure (bar)
      pd: Downstream Pressure (bar)
      masl: elevation (metres above sea level)
      temp: Water temperature (Celsius)
      gravity: standard acceleration due to gravity (m/s2)
    '''
    super().__init__(tempC)
    self.dn = dn
    self.flow = flow
    self.pu = pu
    self.pd = pd
    self.masl = masl
    self.gravity = 9.80665
  
  def atm_pressure(self) -> float:
    '''
    Description: Calculate the Atm. Pressure
    Returns: Atm. Pressure in bar
    '''
    p_at = (1/1000)*((44331.514-self.masl)/11880.516)**(1/0.1902632)
    return p_at

  def velocity(self) -> float:
    '''
    Description: This function calculates the velocity of the fluid in a 
      circular pipe.
    Returns: velocity in meter per second (m/s)
    '''
    v = (self.flow)/((math.pi*self.dn**2)/4)
    return v

  def flow_coefficient(self)->float:
    '''
    Description: 
    Returns:
    '''
    rd = self.relative_density()
    kv = (self.flow*3600)*math.sqrt(rd/(self.pu-self.pd))
    return kv

  def zeta_value(self)-> float:
    '''
    Description: Calculate the resistance Coefficient Zeta in function of Kv
    Returns: zeta (resistance Coefficient)
    '''
    kv = self.flow_coefficient()
    zeta = (1/626.3)*((self.dn*1000)**2/kv)**2
    return zeta

  def sigmas(self)-> float:
    '''
    Description:
    Returns:
    '''
    pv = self.p_vapour()
    p1 = self.pu + self.atm_pressure()
    p2 = self.pd + self.atm_pressure()
    dp = p1-p2
    v2_2g = self.velocity()**2/(2*self.gravity)
    sigma_0 = (p1-pv)/dp 
    sigma_1 = (p2-pv)/dp 
    sigma_2 = (p2-pv)/(dp+v2_2g)
    return (sigma_0, sigma_1, sigma_2)
