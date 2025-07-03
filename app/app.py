from flask import Flask, jsonify, request
from sqlalchemy import create_engine
import pandas as pd
import pickle
import json
from datetime import datetime

with open('model_for_api.pkl', 'rb') as archivo:
    model = pickle.load(archivo)
engine = create_engine("postgresql://postgres:postgres@35.205.146.144/postgres")

app = Flask(__name__)    
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>House Predictions</h1><p>Predice el precio de una vivienda</p>"

# 1. HACER UNA PREDICCION
# Necesitamos el area, numero de dormitorios y numero de baños
# La guardamos en un dataframe 

@app.route('/predict/<string:data>', methods=['GET'])
def predict(data):
        data = json.loads(data)
        prediction = model.predict(data)
        # Creamos un diccionario que contenga el timestamp, los datos y la prediccion
        table = pd.DataFrame(
             {
                  'timestamp': datetime.now(),
                  'data':data,
                  'prediction': prediction
             }
        )

        table.to_sql('predictions', con=engine, index = None, if_exists='append')


        return str(prediction)


@app.route('/predict/history', methods=['GET'])
def history():
      
      query = '''
        SELECT * FROM predictions
'''
      history_predictions = pd.read_sql(query, con=engine)


      return history_predictions.to_dict(orient = 'records')



app.run(host='0.0.0.0', port=5000, debug = True)