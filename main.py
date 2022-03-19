from fastapi import FastAPI
from pydantic import BaseModel
# from sklearn.preprocessing import StandardScaler
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier

import pickle
import numpy as np

# Load model and scaler
loaded_nn1 = pickle.load(open('ml_models/nn1.sav', 'rb'))
loaded_nn2 = pickle.load(open('ml_models/nn2.sav', 'rb'))
loaded_dtree = pickle.load(open('ml_models/dtree.sav', 'rb'))
loaded_rf = pickle.load(open('ml_models/rf.sav', 'rb'))
loaded_svm = pickle.load(open('ml_models/svm.sav', 'rb'))
loaded_logreg = pickle.load(open('ml_models/logreg.sav', 'rb'))
loaded_nb = pickle.load(open('ml_models/nb.sav', 'rb'))

scaler = pickle.load(open('ml_models/scaler.sav', 'rb'))

weather_cond = ['Clear', 'Cloud', 'Sunny', 'Rainy']
# End of model and stuff

class WeatherData(BaseModel):
    temperature: float
    wind_speed: float
    pressure: int
    humidity: int
    vis_km: float
    cloud: int


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/classify/")
def classify(weather_data: WeatherData):
    X = np.array([[weather_data.temperature, weather_data.wind_speed, weather_data.pressure,
                 weather_data.humidity, weather_data.vis_km, weather_data.cloud]])
    X = scaler.transform(X)
    tmp_arr = [0,0,0,0]
    nn1_arr = loaded_nn1.predict(X)        
    nn2_arr = loaded_nn2.predict(X)       
    dtree_arr = loaded_dtree.predict(X)  
    rf_arr = loaded_rf.predict(X)          
    svm_arr = loaded_svm.predict(X)        
    logreg_arr = loaded_logreg.predict(X)  
    nb_arr = loaded_nb.predict(X)   
     
    # ensample process     
    if nn1_arr == 0:
        tmp_arr[0] += 0.9
    elif nn1_arr == 1:
        tmp_arr[1] += 0.9
    elif nn1_arr == 2:
        tmp_arr[2] += 0.9
    elif nn1_arr == 3:
        tmp_arr[3] += 0.9

    if nn2_arr == 0:
        tmp_arr[0] += 0.9
    elif nn2_arr == 1:
        tmp_arr[1] += 0.9
    elif nn2_arr == 2:
        tmp_arr[2] += 0.9
    elif nn2_arr == 3:
        tmp_arr[3] += 0.9

    if dtree_arr == 0:
        tmp_arr[0] += 0.9
    elif dtree_arr == 1:
        tmp_arr[1] += 0.9
    elif dtree_arr == 2:
        tmp_arr[2] += 0.9
    elif dtree_arr == 3:
        tmp_arr[3] += 0.9
  
    if rf_arr == 0:
        tmp_arr[0] += 1
    elif rf_arr == 1:
        tmp_arr[1] += 1
    elif rf_arr == 2:
        tmp_arr[2] += 1
    elif rf_arr == 3:
        tmp_arr[3] += 1
  
    if svm_arr == 0:
        tmp_arr[0] += 0.7
    elif svm_arr == 1:
        tmp_arr[1] += 0.7
    elif svm_arr == 2:
        tmp_arr[2] += 0.7
    elif svm_arr == 3:
        tmp_arr[3] += 0.7
  
    if logreg_arr == 0:
        tmp_arr[0] += 0.7
    elif logreg_arr == 1:
        tmp_arr[1] += 0.7
    elif logreg_arr == 2:
        tmp_arr[2] += 0.7
    elif logreg_arr == 3:
        tmp_arr[3] += 0.7
  
    if nb_arr == 0:
        tmp_arr[0] += 0.6
    elif nb_arr == 1:
        tmp_arr[1] += 0.6
    elif nb_arr == 2:
        tmp_arr[2] += 0.6
    elif nb_arr == 3:
        tmp_arr[3] += 0.6

    if tmp_arr[0]== max(tmp_arr):
        res = 0
    elif tmp_arr[1]== max(tmp_arr):
        res = 1
    elif tmp_arr[2]== max(tmp_arr):
        res = 2
    elif tmp_arr[3]== max(tmp_arr):
        res = 3 

    return {'Condition': weather_cond[res]}