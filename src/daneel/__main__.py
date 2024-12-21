import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *
import os
import pandas as pd
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        nargs='+',
        required=True,
        help="Input par file to pass",
    )

    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        required=False,
        help="Initialise detection algorithms for Exoplanets",
        type=str
        #Previously there was a store true action, why?
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        required=False,
        help="Atmospheric Characterisazion from input transmission spectrum",
        type=str

        #removed action true
    )

    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        required=False,
        help="Create transit lightcurve",
        action="store_true",
    )
    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    #input_pars = Parameters(args.input_file).params

    print("ciao",args.input_file[0][0])


    if args.transit:
        transit = Transit()
        transit.get_transit(args.input_file)
    if "svm"==args.detect:
        print("Starting SVM")
        svm=SVM()
        svm.detect(args.input_file[0])
    if "nn"==args.detect:
        print("Starting NN")
        NN=NNObj()
        NN.LoadAndTrain(args.input_file[0])

    if args.atmosphere=="retrieve":
        with open(args.input_file[0],"r") as f:
                ParamsDict=yaml.safe_load(f)
        inputfile,outputfile,savefile=ParamsDict["PlanetParam"],ParamsDict["output"],ParamsDict["Save"]
        StrCommand="taurex -i "+inputfile+" -o "+outputfile+" -S "+savefile+" --plot --retrieval"
        os.system(StrCommand)
    if args.atmosphere=="model":
        with open(args.input_file[0],"r") as f:
                ParamsDict=yaml.safe_load(f)
        inputfile,outputfile,savefile=ParamsDict["PlanetParam"],ParamsDict["output"],ParamsDict["Save"]
        StrCommand="taurex -i "+inputfile+" -o "+outputfile+" -S "+savefile+" --plot"
        subprocess.run(["taurex","-i",inputfile,"-o",outputfile,"-S",savefile,"--plot"])
        print("Does it wait?")
        Df=pd.read_csv(savefile)
        Df=Df.iloc[:,:-2]
        Df['C'] = np.sqrt(Df.iloc[:, 1])
        df.columns=["wavelength in micron","(rp/rs)^2","sqrt((rp/rs)^2)"]
        Df.to_csv(savefile, sep=" ", index=False)




    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
