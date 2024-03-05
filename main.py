import json
import dill
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
#from pipe import get_coordinates
#csv = get_coordinates.csv

filename = './models/pipe.pkl'
with open(filename, 'rb') as file:
   model = dill.load(file)

class Form(BaseModel):
    session_id: str
    client_id: Optional[str]
    visit_date: Optional[str]
    visit_time:Optional[str]
    visit_number: Optional[int]
    utm_source: Optional[str]
    utm_medium: Optional[str]
    utm_campaign: Optional[str]
    utm_adcontent: Optional[str]
    utm_keyword: Optional[str]
    device_category: Optional[str]
    device_os: Optional[str]
    device_brand:Optional[str]
    device_model: Optional[str]
    device_screen_resolution: Optional[str]
    device_browser: Optional[str]
    geo_country: Optional[str]
    geo_city:  Optional[str]


class Prediction(BaseModel):
    session_id: str
    target: int

app = FastAPI()

@app.get('/status')
def status():
    return 'alive and well'

@app.get('/version')
def version():
   return model['metadata']

@app.post('/predict')
def predict(form:Form):
    x = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(x)
    return {
         'session_id': form.session_id,
         'target_action': y.tolist()[0],

    }



