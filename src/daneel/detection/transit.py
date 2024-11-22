import batman
import numpy as np
from matplotlib import pyplot as plt
import yaml

class Transit:
    def __init__(self):
        pass

    def get_transit(self, PathInputVector):
        Labels=["t0","per","rp","a","inc","ecc","w","limb_dark","u"]
        Names=""
        for PathInput in PathInputVector:
            with open(PathInput,"r") as f:
                ParamsDict=yaml.safe_load(f)
            params = batman.TransitParams()       #object to store transit parameters
            for label in Labels:
                setattr(params,label,ParamsDict[label])    
            t = np.linspace(-0.12, 0.12, 30000)  #times at which to calculate light curve

            m = batman.TransitModel(params, t, max_err = 0.1)    #initializes model
            
            #print(ParamsDict)
            # light curve model
            flux = m.light_curve(params) #calculates light curve
            plt.plot(t, flux, label = Names) # DA MODIFICARE PERCHE' NE ESCE SOLO UNO
            plt.xlabel("Time from central transit [s]")
            plt.ylabel("Relative flux []")
            #plt.yscale('log')
            plt.tight_layout()
            plt.legend(loc='lower left')
            Names=Names+" - "+ParamsDict["Name"]
        plt.title(Names)
        plt.savefig(Names+'.png')