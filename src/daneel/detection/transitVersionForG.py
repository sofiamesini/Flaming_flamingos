import batman
import numpy as np
from matplotlib import pyplot as plt
import yaml
import argparse

def transit(PathInput):
    Labels=["t0","per","rp","a","inc","ecc","w","limb_dark","u"]
    with open(PathInput,"r") as f:
        ParamsDict=yaml.safe_load(f)
    params = batman.TransitParams()       #object to store transit parameters
    for label in Labels:
        setattr(params,label,ParamsDict[label])    
    t = np.linspace(-0.1, 0.1, 2000)  #times at which to calculate light curve

    m = batman.TransitModel(params, t)    #initializes model

    # light curve model
    flux = m.light_curve(params) #calculates light curve
    plt.plot(t, flux)
    plt.xlabel("Time from central transit [s]")
    plt.ylabel("Relative flux []")
    plt.title(ParamsDict["Name"])
    plt.savefig(ParamsDict["Name"]+'.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the transit light curve.")
    parser.add_argument("-i", help="Path parameters", required=True, type=str)
    parser.add_argument("-t", "--transit", help="Generate transit light curve", required=True)    
    args = parser.parse_args()

    if args.transit:
        transit(args.i)  # Path to the YAML parameters file
        
