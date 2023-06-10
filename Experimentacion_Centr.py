from itertools import combinations
import random

import os
import math

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

from logging import INFO
from csv import writer


from numpy.random import seed
from tensorflow.keras.utils import set_random_seed

from tensorflow.keras.metrics import Precision, Recall, TrueNegatives, TruePositives, FalsePositives, FalseNegatives

import regex as re

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
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])
    
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size = 32, epochs = 150, es = True):
    
    callbacks = []
    if es:
        callbacks.append(EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=20, restore_best_weights=True))

    hist = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks = callbacks)
    
    return hist

for n_desconocidos in range(14,0,-1):
    combs = generar_combinaciones(21-n_desconocidos)
    if (n_desconocidos > 1):
        combs = subset_combinaciones_validas(combs, 30)
    
    # Se crea el fichero si no existe previamente
    if not os.path.exists(f"./results/experimentacion/Centralizado/{n_desconocidos}UCs.csv"):
        results_cent = pd.DataFrame(columns=["UCs", "k_acc", "k_sens", "k_spec", "k_f1", "u_acc", "u_sens", "u_spec", "u_f1"])
        results_cent.to_csv(f"./results/experimentacion/Centralizado/{n_desconocidos}UCs.csv", mode='w', index=False, header=True)
    
    # Se busca si ya hay algun registro
    with open(f"./results/experimentacion/Centralizado/{n_desconocidos}UCs.csv", 'r') as f:
        start_comb = len(f.readlines())-1 # restandole el header
        
    # Se comienza/retoma el experimento de n_desconocidos
    for comb in combs[start_comb:]:
        if (uno_por_empresa(comb)):
            global UNSEEN_CLIENTS
            UNSEEN_CLIENTS = list(set(ALL_IDS)-set(comb))
            
            global KNOWN_CLIENTS
            KNOWN_CLIENTS = list(comb)
            
            print(f'{n_desconocidos} - {UNSEEN_CLIENTS}')
            
            # Entrenar modelo centralizado
            X_train, X_val, y_train, y_val = cargar_dataset_varios_clientes(KNOWN_CLIENTS)
            
            seed(1)
            set_random_seed(2)

            # Entrenamiento de modelo centralizado
            model = get_model()
            hist = hist = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=20,
                verbose=0,
                use_multiprocessing=True,
                workers=6,
                validation_data=(X_val, y_val))

            # Evaluar para los clientes conocidos
            res = hist.model.evaluate(X_val, y_val, verbose=2)
            
            tp_k = res[2]
            tn_k = res[3]
            fp_k = res[4]
            fn_k = res[5]
            
            k_acc = (tp_k+tn_k)/(tp_k+tn_k+fp_k+fn_k)
            k_sens = (tp_k)/(tp_k+fn_k)
            k_spec = (tn_k)/(tn_k+fp_k)
            k_f1 = (tp_k)/( tp_k + (fp_k+fn_k)/2 )

            # Evaluar para el cliente nuevo
            X_train, X_val, y_train, y_val = cargar_dataset_varios_clientes(UNSEEN_CLIENTS)
            res = hist.model.evaluate(X_val, y_val, verbose=2)

            tp_u = res[2]
            tn_u = res[3]
            fp_u = res[4]
            fn_u = res[5]

            u_acc = (tp_u+tn_u)/(tp_u+tn_u+fp_u+fn_u)
            u_sens = (tp_u)/(tp_u+fn_u)
            u_spec = (tn_u)/(tn_u+fp_u)
            u_f1 = (tp_u)/( tp_u + (fp_u+fn_u)/2 )
            
            results_cent = pd.DataFrame(columns=["UCs", "k_acc", "k_sens", "k_spec", "k_f1", "u_acc", "u_sens", "u_spec", "u_f1"])
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
            
            results_cent = results_cent.append(fed_res, ignore_index=True)
            results_cent.to_csv(f"./results/experimentacion/Centralizado/{n_desconocidos}UCs.csv", mode='a', index=False, header=False)