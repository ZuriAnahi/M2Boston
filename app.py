from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from scikeras.wrappers import KerasRegressor

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Definir la función de creación del modelo
def create_model():
    model = Sequential()
    model.add(Input(shape=(6,)))  # Ajustar el número de características de entrada
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Cargar el modelo entrenado
try:
    model = joblib.load('modeloBoston.pkl')
    app.logger.debug('Modelo cargado correctamente.')
except Exception as e:
    app.logger.error(f'Error al cargar el modelo: {str(e)}')
    model = None

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado correctamente.'}), 400

    try:
        # Obtener los datos enviados en el request
        NOX = float(request.form['NOX'])
        RM = float(request.form['RM'])
        DIS = float(request.form['DIS'])
        PTRATIO = float(request.form['PTRATIO'])
        B = float(request.form['B'])
        LSTAT = float(request.form['LSTAT'])

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[NOX, RM, DIS, PTRATIO, B, LSTAT]], columns=['NOX', 'RM', 'DIS', 'PTRATIO', 'B', 'LSTAT'])

        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver la predicción como respuesta JSON
        return jsonify({'categoria': float(prediction[0])})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
