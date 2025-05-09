


- make a python program that calculates the position of laser beams after propagation through air

- the source is a fan of laser beams with a given angular separation. the fan of rays is spread out in the veritcal direction. 

- a function called temp_air describes the temperature of the air as a function of height above the ground. the function initially is a constant value of 20 degrees Celsius, but it can be modified to include a linear or a polynomial temperature gradient. The function should take the height as an input and return the temperature in degrees Celsius or Kelvin.

- the function should also include a parameter for the temperature at sea level, which can be modified to simulate different atmospheric conditions.

- one suggestion is to: do calculations of the refractive index of air as a function of temperature and pressure,  Ray Equation in a Medium with Variable Refractive Index,  Approximation in Stratified Atmosphere and do a numerical integration of the ray equation 

- a graph should show the position of the laser beams after propagation through air. 


variables: 
dist: distance from the laser to the wall, start value 500m
n_beams: number of laser beams, start value 10
d_angle: angular separation between beams, starf value 0.1 mrad
w_laser: wavelength of the laser, start value 660 nm
d_step: step size for the propagation, start value 0.1 m
h_limit: rays above this height are not considered, 30 m 


-------


- add a variable called h_start that represents the height of the laser beams at the start of the propagation. The default value is 2 m, but it can be modified to simulate different initial heights.

- terminate beams that reach the ground or the wall.

- add a graph that shows the air temperature as a function of height. The graph should include the temperature at sea level and the temperature at the height of the laser beams. The graph should also include a line that represents the temperature gradient, which can be modified to simulate different atmospheric conditions.

- add a graph that shows the difference in landing height between adjecent beams. The graph should include a line that represents the average difference in landing height between adjacent beams

---

add dashed thin lines that shows the beam trajectories for the case that the air is at a constant temperature of 20 degrees Celsius. The lines should be in a the same color as the beams, but with a lower opacity and dashed.