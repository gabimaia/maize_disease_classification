"""
Creator: Gabriel Maia
Date: May 2022
Create API
"""
# from typing import Union
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import pandas as pd
import joblib
import os
import wandb
import sys
from source.api.pipeline import FeatureSelector, CategoricalTransformer, NumericalTransformer

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)
setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# name of the model artifact
artifact_model_name = "Decision_tree_heart_disease/model_export:latest"

# initiate the wandb project
run = wandb.init(project="Decision_tree_heart_disease",job_type="api")

# create the api
app = FastAPI()

# declare request example data using pydantic
# a person in our dataset has the following attributes
class Person(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str	
    MaxHR: int
    ExerciseAngina: str	
    Oldpeak: float	
    ST_Slope: str

    class Config:
        schema_extra = {
            "example": {
                "Age": 72,
                "Sex": 'M',
                "ChestPainType": 'NAP',
                "RestingBP": 120,
                "Cholesterol": 304,
                "FastingBS": 1,
                "RestingECG": 'Normal',
                "MaxHR": 120,
                "ExerciseAngina": 'Y',
                "Oldpeak": 1.0,
                "ST_Slope": 'Up'
            }
        }
  

def predict(image: Image.Image):
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    result = decode_predictions(model.predict(image), 2)[0]
    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        response.append(resp)
    return response


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

# give a greeting using GET
@app.get('/index')
async def hello_world():
    return "hello world"

# run the model inference and use a Person data structure via POST to the API.
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction




