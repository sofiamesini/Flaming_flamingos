import batman
import numpy as np
import matplotlib as plt

params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 1.338231602                #orbital period
params.rp = (1.3846*69911)/(1.15 *695000)  #planet radius (in units of stellar radii)
params.a = (0.02312*149597871)/(1.15*695000)  #semi-major axis (in units of stellar radii)
params.inc = 88.87                      #orbital inclination (in degrees)
params.ecc = 0.0053                    #eccentricity
params.w = 90.                        #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.46976666666666667, 0.18760000000000002]      #limb darkening coefficients [u1, u2, u3, u4]

t = np.linspace(-0.025, 0.025, 1000)  #times at which to calculate light curve
m = batman.TransitModel(params, t)    #initializes model

# light curve model
flux = m.light_curve(params) #calculates light curve
plt.plot(t, flux)
plt.xlabel("Time from central transit")
plt.ylabel("Relative flux")
plt.savefig('WASP-4 b_assignment1_taskF.png')
