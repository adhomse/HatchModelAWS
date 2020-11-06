from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn import tree
from sklearn.externals import joblib

import pickle
# from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    #*****************
    file = os.path.join(args.train, "GameWiseAttemptsData.csv")
    dataset = pd.read_csv(file, engine="python")
    PredictNumberOfAttempts = dataset[['id_org','child_id','user_id','domain_id','subdomain_id','game_id','Skill Level ID','NumberOfTimesGamePlayed','gender']]
    print(PredictNumberOfAttempts.shape)
    PredictNumberOfAttempts = PredictNumberOfAttempts.dropna()
    #Remove outlier
    PredictionAlgoDFRemOut= PredictNumberOfAttempts.loc[PredictNumberOfAttempts['NumberOfTimesGamePlayed']<6] 
    
    Y= PredictionAlgoDFRemOut['NumberOfTimesGamePlayed']
    del PredictionAlgoDFRemOut['NumberOfTimesGamePlayed']
    InputFeatures = PredictionAlgoDFRemOut.values
    Y = Y.values
    GBR = GradientBoostingRegressor(n_estimators=140, max_depth=3,verbose=2)
    X_train, X_test , Y_train , Y_test =train_test_split(InputFeatures, Y , test_size = 0.10,random_state =2)
    
    GBR.fit(X_train, Y_train)
    print(GBR.feature_importances_)
    print("Accuracy on training data --> ", GBR.score(X_train, Y_train)*100)
    
    print("Accuracy --> ", GBR.score(X_test, Y_test)*100)
    joblib.dump(GBR, os.path.join(args.model_dir, "model.joblib"))
    #print('X_test',X_test)
    # values = 
    #PredictedpriceGBR = GBR.predict(X_test)
    #print(PredictedpriceGBR)
    #pickle.dump(GBR, open("attemptPredict.pickle.dat", "wb"))
    # predictionValue = GBR.predict([[13609.0,927514.0,709021.0,4.0,5.0,2.0,1.0,2.0]])
    # print(predictionValue)
    
    
    
    #****************


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    regressor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return regressor
