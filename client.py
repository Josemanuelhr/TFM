import os
import math
import sys

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adadelta

from typing import Dict

from flwr.common.logger import log
from logging import INFO
from csv import writer


from numpy.random import seed
from tensorflow.keras.utils import set_random_seed

import matplotlib.pyplot as plt

import os

def get_empresa_conductor(conductor_id):
    for i in range(1,4):
        if (conductor_id in CONDUCTORES_IDS[i]):
            return i
        
def prepare_model_data(client_file):
    df = pd.read_csv(client_file)
    
    train, test = train_test_split(df, test_size=0.30, random_state=42)
    
    X_train = train[['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma','eog_blinks', 'eog_var']]
    X_test = test[['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma','eog_blinks', 'eog_var']]
    y_train = train['y_class']
    y_test = test['y_class']
    
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def cargar_dataset_varios_clientes(clientes):
    base_path = "./data/centralizado"
    
    X_train, X_val, y_train, y_val = prepare_model_data(f'{base_path}/cliente_{clientes[0]}.csv')
    
    for cid in clientes[1:]:
        path = f'{base_path}/cliente_{cid}.csv'
        X_train_act, X_val_act, y_train_act, y_val_act = prepare_model_data(path)
    
        X_train = np.vstack((X_train, X_train_act))
        X_val = np.vstack((X_val, X_val_act))
        y_train = np.concatenate((y_train, y_train_act))
        y_val = np.concatenate((y_val, y_val_act))
        
    return X_train, X_val, y_train, y_val

def get_model():
    # Model best hyperparameters (Ver notebook Hito0-Optimizacion-Baseline)
    neurons = 36
    activation = "relu"
    learning_rate = 0.180165
    optimizer = Adadelta(learning_rate=learning_rate)
    
    input_shape = (7,)
    
    # Create model
    model = Sequential()
    
    model.add(Dense(neurons, input_shape=input_shape, activation=activation))
    
    model.add(BatchNormalization())
        
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def get_weights_from_file(path):
    a = np.load(path, allow_pickle=True)

    n_elems = [252, 36,
               36, 36, 36, 36,
               1296, 36,
               1296, 36,
               1296, 36,
               36, 1]

    weights = []

    # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#version-numbering
    # En base a la doc sabemos que los datos de interes estan al final, y como deben ser float32 => 4 bytes
    # por lo que se toman los n_elementos*4bytes del final

    for i, t in enumerate(a["arr_0"][0].tensors):
        act = np.frombuffer(t[-n_elems[i]*4:], dtype=np.float32)
        weights.append(act)

    # Se cambia la forma para que se adapte a la del modelo
    weights[0] = weights[0].reshape(7,36)
    weights[6] = weights[6].reshape(36,36)
    weights[8] = weights[8].reshape(36,36)
    weights[10] = weights[10].reshape(36,36)
    weights[12] = weights[12].reshape(36,1)
    
    return weights

''' CONFIG L1 '''
class ConductorClient(fl.client.NumPyClient):
    def __init__(self, cid, empresa, model, x_train, y_train, x_val, y_val) -> None:
        self.cid = cid
        self.empresa = empresa
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        seed(1)
        set_random_seed(2)
        
        self.model.fit(self.x_train, self.y_train,
                       epochs=1,
                       batch_size=32,
                       verbose=0)
        
        return self.model.get_weights(), len(self.x_train), {"client": self.cid, "empresa": self.empresa}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        
        return loss, len(self.x_val), {"accuracy": acc, "client": self.cid, "empresa": self.empresa}
    
def conductor_fn(cid: str) -> fl.client.Client:
    model = get_model()
    
    empresa = get_empresa_conductor(int(cid))
    
    # Load data partition
    base_path = "./data/horizontal_v3/"
    path = f"{base_path}empresa_{empresa}/cliente_{cid}.csv"
    
    x_train_cid, x_val_cid, y_train_cid, y_val_cid = prepare_model_data(path)

    # Create and return client
    print(f"Creado conductor {cid}")
    return ConductorClient(cid, empresa, model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)

class L1SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        # COMPROBAR DE QUE EMPRESA ES EL CLIENTE
        emp_id = results[1].metrics["empresa"]
        
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"./rounds/hito3_L1-round-{rnd}-E{emp_id}-weights.npz", aggregated_weights)
            
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        super_result = super().aggregate_evaluate(rnd, results, failures)
        
        return super_result

''' CONFIGURACION DEL CLIENTE FEDERADO (L2) '''
class EmpresaClient(fl.client.NumPyClient):
    def __init__(self, cid, model, x_train, y_train, x_val, y_val) -> None:
        self.cid = cid
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        emp_id = int(self.cid[-1])
        
        def L1_fit_config(rnd: int) -> Dict[str, str]:
            config = {
                "round": str(rnd)
            }
            return config
        
        fl.simulation.start_simulation(
        client_fn=conductor_fn,
        clients_ids=CONDUCTORES_IDS[emp_id],
        client_resources={"num_cpus": 2},
        num_rounds=1,
        strategy=L1SaveModelStrategy(
                min_available_clients = len(CONDUCTORES_IDS[emp_id]),
                min_fit_clients = len(CONDUCTORES_IDS[emp_id]),
                min_eval_clients = len(CONDUCTORES_IDS[emp_id]),
                on_fit_config_fn = L1_fit_config,
                on_evaluate_config_fn = L1_fit_config,
                accept_failures=False,
                initial_parameters=parameters
            )
        )
        
        # Se cargan los pesos agregados para la epoch de todos los conductores
        rnd = int(config["round"])
        path = f"./rounds/hito3_L1-round-{1}-E{emp_id}-weights.npz"
        
        weights = get_weights_from_file(path)
        
        # Se recuentan cuantas observaciones de train hay
        obs_total = len(self.x_train)
        
        return weights, obs_total, {"client": self.cid}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        
        # Se recuentan cuantas observaciones de test hay
        obs_total = len(self.x_val)
        
        return loss, obs_total, {"accuracy": acc, "client": self.cid}
    

''' EJECUCION DEL CLIENTE '''
# sys.argv[1] = empresa_X
# sys.argv[2] = puerto del servidor
def main():
    global EMPRESAS_IDS
    EMPRESAS_IDS = ["empresa_1", "empresa_2", "empresa_3"]

    global CONDUCTORES_IDS
    CONDUCTORES_IDS = {
        1: [3,4,5,12,  2,8,  1],
        2: [13,14,    6,7,10,11,16],
        3: [15,18,  9,17,20,21,  19]
    }

    model = get_model()

    emp_id = int(sys.argv[1][-1])
    x_train_cid, x_val_cid, y_train_cid, y_val_cid = cargar_dataset_varios_clientes(CONDUCTORES_IDS[emp_id])

    fl.client.start_numpy_client(
        server_address="localhost:"+sys.argv[2],
        client=EmpresaClient(sys.argv[1], model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)
    )

if __name__ == "__main__":
    main()