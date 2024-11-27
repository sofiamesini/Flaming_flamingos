import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC, NuSVC
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
import os
import yaml


class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        return np.abs(fft.fft(np.array(X), n=X.size))

    def process(self, df_train_x, df_dev_x):
        # Apply fourier transform
        df_train_x.dropna(inplace=True)
        df_dev_x.dropna(inplace=True)
        if self.fourier:
            print("Applying Fourier...")
            shape_train = df_train_x.shape
            shape_dev = df_dev_x.shape
            df_train_x = df_train_x.apply(self.fourier_transform,axis=1)
            df_dev_x = df_dev_x.apply(self.fourier_transform,axis=1)

            df_train_x_build = np.zeros(shape_train)
            df_dev_x_build = np.zeros(shape_dev)

            for ii, x in enumerate(df_train_x):
                df_train_x_build[ii] = x

            for ii, x in enumerate(df_dev_x):
                df_dev_x_build[ii] = x

            df_train_x = pd.DataFrame(df_train_x_build)
            df_dev_x = pd.DataFrame(df_dev_x_build)


            # Keep first half of data as it is symmetrical after previous steps
            df_train_x = df_train_x.iloc[:,:(df_train_x.shape[1]//2)].values
            df_dev_x = df_dev_x.iloc[:,:(df_dev_x.shape[1]//2)].values

        # Normalize
        if self.normalize:
            print("Normalizing...")
            df_train_x = pd.DataFrame(normalize(df_train_x))
            df_dev_x = pd.DataFrame(normalize(df_dev_x))

            # df_train_x = df_train_x.div(df_train_x.sum(axis=1), axis=0)
            # df_dev_x = df_dev_x.div(df_dev_x.sum(axis=1), axis=0)


        # Gaussian filter to smooth out data
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.gaussian_filter(df_train_x, sigma=10)
            df_dev_x = ndimage.gaussian_filter(df_dev_x, sigma=10)

        if self.standardize:
            # Standardize X data
            print("Standardizing...")
            std_scaler = StandardScaler()
            df_train_x = std_scaler.fit_transform(df_train_x)
            df_dev_x = std_scaler.transform(df_dev_x)

        print("Finished Processing!")
        return df_train_x, df_dev_x



class SVM:
    def __init__(self):
        pass

    def LoadData(self,root_dir):
        train_dataset_path = os.path.join(root_dir, "exoTrain.csv")
        dev_dataset_path = os.path.join(root_dir, "exoTest.csv")

        print("Loading datasets...")
        df_train = pd.read_csv(train_dataset_path, encoding = "ISO-8859-1")
        df_dev = pd.read_csv(dev_dataset_path, encoding = "ISO-8859-1")
        print("Loaded datasets!")

        # Generate X and Y dataframe sets
        df_train_x = df_train.drop('LABEL', axis=1)
        df_dev_x = df_dev.drop('LABEL', axis=1)
        df_train_y = df_train.LABEL
        df_dev_y = df_dev.LABEL
        return df_train_x,df_dev_x,df_train_y,df_dev_y
    def detect(self,ConfigFile):
        with open(ConfigFile,"r") as f:
                ParamsDict=yaml.safe_load(f)
        KernelUsed,root_dir=ParamsDict["kernel"],ParamsDict["dir"]
        df_train_x,df_dev_x,df_train_y,df_dev_y=self.LoadData(root_dir)
        def np_X_Y_from_df(df):
            df = shuffle(df)
            df_X = df.drop(['LABEL'], axis=1)
            X = np.array(df_X)
            Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']),1))
            Y = Y_raw == 2
            return X, Y
        
        # Process dataset
        LFP = LightFluxProcessor(
        fourier=True,
        normalize=True,
        gaussian=True,
        standardize=True)
        df_train_x, df_dev_x = LFP.process(df_train_x, df_dev_x)

        # display(df_train_x)


        # Rejoin X and Y
        df_train_processed = pd.DataFrame(df_train_x).join(pd.DataFrame(df_train_y))
        df_dev_processed = pd.DataFrame(df_dev_x).join(pd.DataFrame(df_dev_y))

        # Load X and Y numpy arrays
        X_train, Y_train = np_X_Y_from_df(df_train_processed)
        X_dev, Y_dev = np_X_Y_from_df(df_dev_processed)
        model = SVC(kernel=KernelUsed,degree=4)


        X_train_sm, Y_train_sm = X_train, Y_train

        # Train
        print("Training...")
        model.fit(X_train_sm, Y_train_sm)

        train_outputs = model.predict(X_train_sm)
        dev_outputs = model.predict(X_dev)
        print("Finished Training!")
        train_outputs = model.predict(X_train_sm)
        dev_outputs = model.predict(X_dev)
        train_outputs = np.rint(train_outputs)
        dev_outputs = np.rint(dev_outputs)
        accuracy_train = accuracy_score(Y_train_sm, train_outputs)
        accuracy_dev = accuracy_score(Y_dev, dev_outputs)
        precision_train = precision_score(Y_train_sm, train_outputs)
        precision_dev = precision_score(Y_dev, dev_outputs)
        recall_train = recall_score(Y_train_sm, train_outputs)
        recall_dev = recall_score(Y_dev, dev_outputs)
        confusion_matrix_train = confusion_matrix(Y_train_sm, train_outputs)
        confusion_matrix_dev = confusion_matrix(Y_dev, dev_outputs)
        classification_report_train = classification_report(Y_train_sm, train_outputs)
        classification_report_dev = classification_report(Y_dev, dev_outputs)
        print(" ")
        print(" ")
        print("Yes INJECTION","Kernel: ", KernelUsed)
        print("Train Set Error", 1.0 - accuracy_train)
        print("Dev Set Error", 1.0 - accuracy_dev)
        print("------------")
        print("Precision - Train Set", precision_train)
        print("Precision - Dev Set", precision_dev)
        print("------------")
        print("Recall - Train Set", recall_train)
        print("Recall - Dev Set", recall_dev)
        print("------------")
        print("Confusion Matrix - Train Set")
        print(confusion_matrix_train)
        print("Confusion Matrix - Dev Set")
        print(confusion_matrix_dev)
        print("------------")
        print(" ")
        print(" ")
        print("------------")
        print("classification_report_train")
        print(classification_report_train)
        print("classification_report_dev")
        print(classification_report_dev)
            