from itertools import combinations
import random

import os
import math

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

from tensorflow.keras.metrics import Precision, Recall, TrueNegatives, TruePositives, FalsePositives, FalseNegatives

import regex as re

''' UTILS '''
ALL_IDS = list(range(1,21+1))

CONDUCTORES_IDS = {
    1: [3,4,5,12,  2,8,  1],
    2: [13,14,    6,7,10,11,16],
    3: [15,18,  9,17,20,21,  19]
}

# Parametros:
    # tam: numero de sujetos que debe haber en las combinaciones generadas
# Retorna:
    # Lista de listas, cada lista se corresponde con una combinacion diferente de tamaño n_desconocidos
def generar_combinaciones(tam):
    return list(combinations(ALL_IDS, tam))

# Parametros:
    # comb: lista con los conductores de la combinacion
# Retorna:
    # True: si hay al menos uno de cada empresa
    # False: en caso contrario
def uno_por_empresa(comb):
    conds = {
        1: False,
        2: False,
        3: False
    }
    
    for cid in comb:
        for empid in CONDUCTORES_IDS:
            if(cid in CONDUCTORES_IDS[empid]):
                conds[empid] = True
    
    return conds[1] and conds[2] and conds[3]

def subset_combinaciones_validas(combs, n):
    seleccionadas = []
    idx_comprobados = []
    n_seleccionadas = 0
    
    seed(123)
    
    while (n_seleccionadas<n):
        idx = int(np.random.randint(0, len(combs), 1))
        
        if (idx not in idx_comprobados):
            comb = combs[idx]
            
            if (uno_por_empresa(comb)):
                seleccionadas.append(comb)
                n_seleccionadas += 1
                
        idx_comprobados.append(idx)
            
    return seleccionadas


''' FL Jerarquico '''
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

def get_data_empresa(empresa):
    base_path = "./data/horizontal_v3"
    # Cargar y procesar datos de todos sus clientes
    clientes = os.listdir(f"{base_path}/{empresa}/")
    
    for client_id in UNSEEN_CLIENTS:
        try:
            clientes.remove(f'cliente_{client_id}.csv')
        except:
            pass
    
    X_train, X_test, y_train, y_test = prepare_model_data(f'{base_path}/{empresa}/{clientes[0]}')
    
    # Debe funcionar aunq exista solo clientes[0]
    for file in clientes[1:]:
        path = f'{base_path}/{empresa}/{file}'
        X_train_act, X_test_act, y_train_act, y_test_act = prepare_model_data(path)

        X_train = np.vstack((X_train, X_train_act))
        X_test = np.vstack((X_test, X_test_act))
        y_train = np.concatenate((y_train, y_train_act))
        y_test = np.concatenate((y_test, y_test_act))
        
    return X_train, X_test, y_train, y_test

def get_empresa_conductor(conductor_id):
    for i in range(1,4):
        if (conductor_id in CONDUCTORES_IDS[i]):
            return i

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

''' FL: Conductor '''
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
    return ConductorClient(cid, empresa, model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)

class L1SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        # COMPROBAR DE QUE EMPRESA ES EL CLIENTE
        emp_id = results[0][1].metrics["empresa"]
        
        if aggregated_weights is not None:
            # Save aggregated_weights
            np.savez(f"./tmp/hito3_L1-round-{CURRENT_RND}-E{emp_id}-weights.npz", aggregated_weights)
            
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        super_result = super().aggregate_evaluate(rnd, results, failures)
     
        return super_result
    
''' FL: Empresa '''
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
                
        # Se cargan los pesos agregados para la epoch de todos los conductores
        rnd = int(config["round"])
        path = f"./tmp/hito3_L1-round-{CURRENT_RND}-E{emp_id}-weights.npz"
        
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

    
def empresa_fn(cid: str) -> fl.client.Client:
    model = get_model()
    
    # Load data partition    
    x_train_cid, x_val_cid, y_train_cid, y_val_cid = get_data_empresa(cid)
    
    # Create and return client
    return EmpresaClient(cid, model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)

class L2SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_weights is not None:
            # Save aggregated_weights
            np.savez(f"./tmp/hito3_L2-round-{CURRENT_RND}-weights.npz", aggregated_weights)
            
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        super_result = super().aggregate_evaluate(rnd, results, failures)
        
        data = {}
        for r in results:
            acc = r[1].metrics["accuracy"]
            client = r[1].metrics["client"]
            data[client] = acc
            
        df = pd.DataFrame(data, index=[0], columns=sorted(data.keys()))
        df.to_csv(f"./results/experimentacion/UC3/hito3.csv", mode='a', index=False, header=False)
        
        return super_result

''' Experimentacion '''
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

def setup_model(weights_path):
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

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])
    
    a = np.load(weights_path, allow_pickle=True)

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

    model.set_weights(weights)
    
    return model

def evaluate_FLmodel(KCs, UCs, rnd):
    path = f"./tmp/hito3_L2-round-{rnd}-weights.npz"
    model = setup_model(path)
    
    # Evaluar para los clientes conocidos
    X_train, X_val, y_train, y_val = cargar_dataset_varios_clientes(KCs)
    res = model.evaluate(X_val, y_val, verbose=0)
    
    tp_k = res[2]
    tn_k = res[3]
    fp_k = res[4]
    fn_k = res[5]
    
    k_acc = (tp_k+tn_k)/(tp_k+tn_k+fp_k+fn_k)
    k_sens = (tp_k)/(tp_k+fn_k)
    k_spec = (tn_k)/(tn_k+fp_k)
    k_f1 = (tp_k)/( tp_k + (fp_k+fn_k)/2 )
    
    # Evaluar para los clientes nuevos
    X_train, X_val, y_train, y_val = cargar_dataset_varios_clientes(UCs)
    res_c = model.evaluate(X_val, y_val, verbose=0)
    
    tp_u = res_c[2]
    tn_u = res_c[3]
    fp_u = res_c[4]
    fn_u = res_c[5]
    
    u_acc = (tp_u+tn_u)/(tp_u+tn_u+fp_u+fn_u)
    u_sens = (tp_u)/(tp_u+fn_u)
    u_spec = (tn_u)/(tn_u+fp_u)
    u_f1 = (tp_u)/( tp_u + (fp_u+fn_u)/2 )
    
    return k_acc, k_sens, k_spec, k_f1, u_acc, u_sens, u_spec, u_f1

''' Configuracion '''
N_RNDS = 15

EMPRESAS_IDS = ["empresa_1", "empresa_2", "empresa_3"]

# De 1 a 14
for n_desconocidos in range(1,10+1):
    combs = generar_combinaciones(21-n_desconocidos)
    if (n_desconocidos > 1):
        combs = subset_combinaciones_validas(combs, 30)
    
    # Se crea el fichero si no existe previamente
    if not os.path.exists(f"./results/experimentacion/UC3/hito3_{n_desconocidos}UCs.csv"):
        results_fed = pd.DataFrame(columns=["UCs", "k_acc", "k_sens", "k_spec", "k_f1", "u_acc", "u_sens", "u_spec", "u_f1"])
        results_fed.to_csv(f"./results/experimentacion/UC3/hito3_{n_desconocidos}UCs.csv", mode='w', index=False, header=True)
    
    # Se busca si ya hay algun registro
    with open(f"./results/experimentacion/UC3/hito3_{n_desconocidos}UCs.csv", 'r') as f:
        start_comb = len(f.readlines())-1 # restandole el header
        
    # Se comienza/retoma el experimento de n_desconocidos
    for comb in combs[start_comb:]:
        if (uno_por_empresa(comb)):
            global UNSEEN_CLIENTS
            UNSEEN_CLIENTS = list(set(ALL_IDS)-set(comb))
            
            global KNOWN_CLIENTS
            KNOWN_CLIENTS = list(comb)
            
            # Buscar si la ejecucion anterior acabo antes de la ronda = N_RNDS
            r = re.compile("hito3_L2-round-(\d+)-weights.npz")
            base_dir = './tmp/'

            start_rnd = 1
            for elem in list(filter(r.match, os.listdir(base_dir))):
                act_rnd = int(re.search(r, elem).group(1))
                if (act_rnd > start_rnd):
                    start_rnd = act_rnd
            
            # FIX: se empieza SIN sumar uno para repetir la ultima ronda por si dio error

            # Si acabo bien se crea un nuevo fichero para almacenar resultados del entrenamiento
            if (start_rnd == 1):
                tmp_df = pd.DataFrame(columns=["empresa_1", "empresa_2", "empresa_3"])
                tmp_df.to_csv(f"./results/experimentacion/UC3/hito3.csv", mode='w', index=False, header=True)
            
            # Comprobar que el fichero temporal tiene el num_lineas correcto y no quedo nada de la ronda fallida
            with open(f"./results/experimentacion/UC3/hito3.csv", 'r') as f:
                lines = f.readlines()
                n_lines = len(lines)
            if (n_lines > start_rnd): # Si start_rnd=2, deberian estar la 0 header y la 1 (ronda federada 1)
                with open(f"./results/experimentacion/UC3/hito3.csv", 'w') as f:
                    for i in range(0,start_rnd):
                        f.write(lines[i])
            
            # FL
            for fed_rnd in range(start_rnd,N_RNDS+1):
                global CURRENT_RND
                CURRENT_RND = fed_rnd

                if fed_rnd == 1:
                    seed(1)
                    set_random_seed(2)

                    model = get_model()
                    weights = model.get_weights()
                else:
                    weights = get_weights_from_file(f"./tmp/hito3_L2-round-{fed_rnd-1}-weights.npz") # Resultado ronda anterior
                    
                parameters = fl.common.weights_to_parameters(weights)
                
                ''' L1 '''
                for empresa in EMPRESAS_IDS:

                    emp_id = int(empresa[-1])

                    def L1_fit_config(rnd: int) -> Dict[str, str]:
                        config = {
                            "round": str(CURRENT_RND)
                        }
                        return config
                    
                    # Tomar de los KNOWN_CLIENTS los correspondentes a emp_id
                    conductores_emp_actual = []
                    for cond_id in KNOWN_CLIENTS:
                        if (cond_id in CONDUCTORES_IDS[emp_id]):
                            conductores_emp_actual.append(cond_id)
                    
                    fl.simulation.start_simulation(
                        client_fn=conductor_fn,
                        clients_ids=conductores_emp_actual,
                        client_resources={"num_cpus": 6},
                        num_rounds=1,
                        strategy=L1SaveModelStrategy(
                            min_available_clients = len(conductores_emp_actual),
                            min_fit_clients = len(conductores_emp_actual),
                            min_eval_clients = len(conductores_emp_actual),
                            on_fit_config_fn = L1_fit_config,
                            on_evaluate_config_fn = L1_fit_config,
                            accept_failures=False,
                            initial_parameters=parameters
                        )
                    )

                ''' L2 '''
                def L2_fit_config(rnd: int) -> Dict[str, str]:
                    config = {
                        "round": str(CURRENT_RND)
                    }
                    return config

                fl.simulation.start_simulation(
                    client_fn=empresa_fn,
                    clients_ids=EMPRESAS_IDS,
                    client_resources={"num_cpus": 3},
                    num_rounds=1,
                    strategy=L2SaveModelStrategy(
                        min_available_clients = len(EMPRESAS_IDS),
                        min_fit_clients = len(EMPRESAS_IDS),
                        min_eval_clients = len(EMPRESAS_IDS),
                        on_fit_config_fn = L2_fit_config,
                        on_evaluate_config_fn = L2_fit_config,
                        accept_failures=False,
                        initial_parameters=parameters
                    )
                )
            
            # Leer resultados del entrenamiento
            df = pd.read_csv('./results/experimentacion/UC3/hito3.csv')
            df["mean"] = df.mean(numeric_only=True, axis=1)

            best_rnd = df["mean"].idxmax()+1
            
            # Evaluar con los pesos de la mejor ronda tanto para clientes conocidos como desconocidos
            k_acc, k_sens, k_spec, k_f1, u_acc, u_sens, u_spec, u_f1 = evaluate_FLmodel(KNOWN_CLIENTS, UNSEEN_CLIENTS, best_rnd)
            
            # Almacenar resultados de este entrenamiento
            results_fed = pd.DataFrame(columns=["UCs", "k_acc", "k_sens", "k_spec", "k_f1", "u_acc", "u_sens", "u_spec", "u_f1"])
            fed_res = {
                "UCs": UNSEEN_CLIENTS,
                "k_acc": k_acc,
                "k_sens": k_sens,
                "k_spec": k_spec,
                "k_f1": k_f1,
                "u_acc": u_acc,
                "u_sens": u_sens,
                "u_spec": u_spec,
                "u_f1": u_f1
            }
            
            results_fed = results_fed.append(fed_res, ignore_index=True)
            results_fed.to_csv(f"./results/experimentacion/UC3/hito3_{n_desconocidos}UCs.csv", mode='a', index=False, header=False)

            # Se borran los ficheros temporales creados
            base_dir = './tmp/'
            for f in os.listdir(base_dir):
                os.remove(os.path.join(base_dir, f))