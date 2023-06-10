import mne
from mne.externals.pymatreader import read_mat

import neurokit2 as nk

import numpy as np
import pandas as pd

from scipy import stats

from os import listdir, path
import re


''' PSD '''
def get_psd_data(client):
    mat_data = read_mat("./data/SEED-VIG/EEG_Feature_5Bands/" + client)
    
    n_channels = 17
    n_samples = 885
    n_bands = 5
    
    # Se calcula la media de cada banda a lo largo de los 17 canales
    PSD = {
        "psd_delta": np.zeros(n_samples),
        "psd_theta": np.zeros(n_samples),
        "psd_alpha": np.zeros(n_samples),
        "psd_beta": np.zeros(n_samples),
        "psd_gamma": np.zeros(n_samples)
    }

    for s in range(n_samples):
        avg_psd = np.zeros(n_bands)
        for b in range(n_bands):
            for c in range(n_channels):
                avg_psd[b] += mat_data["psd_movingAve"][c][s][b]

        PSD["psd_delta"][s] = avg_psd[0]
        PSD["psd_theta"][s] = avg_psd[1]
        PSD["psd_alpha"][s] = avg_psd[2]
        PSD["psd_beta"][s] = avg_psd[3]
        PSD["psd_gamma"][s] = avg_psd[4]
    
    psd_df = pd.DataFrame.from_dict(PSD)
    
    return psd_df

''' EOG '''
def clean_eog(data, sampling_rate=125, method='neurokit'):
    return nk.eog_clean(data, sampling_rate=sampling_rate, method=method)

def amend_outliers_eog(data):
    data = abs(data)
    thresh=np.percentile(data, 99.5)

    data_o = []
    for el in data:
        if el > thresh:
            el = thresh
        
        data_o.append(el)
    
    return data_o

def make_epochs(eog, samples_per_epoch=1000):
    epochs = []
    for e in range(len(eog)//samples_per_epoch):
        epochs.append(eog[e*samples_per_epoch:(e+1)*samples_per_epoch])
    
    return np.array(epochs)

def eog_epochs_get_features(eog_epochs, thresh):
    eog_blinks = []
    eog_var = []
    
    for e in range(len(eog_epochs)):
        data = eog_epochs[e]
        
        # abs de nuevo por si no se usa el remove_outliers_eog()
        blinks = mne.preprocessing.peak_finder(abs(data), thresh=thresh, extrema=1, verbose=False)

        eog_blinks.append(len(blinks[0]))
        eog_var.append(data.var())
    
    return np.array(eog_blinks), np.array(eog_var)

def get_eog_data(client):
    mat_data = read_mat("./data/SEED-VIG/Raw_Data/" + client)

    eogv = mat_data['EOG']['eog_v']*1e-6
    eogv_clean = clean_eog(eogv)
    eogv_clean_o = amend_outliers_eog(eogv_clean)

    eog_epochs = make_epochs(eogv_clean_o, 1000)

    thresh = np.percentile(eogv_clean_o, 95)
    eog_blinks, eog_var = eog_epochs_get_features(eog_epochs, thresh)
    
    data = {
        "eog_blinks": eog_blinks,
        "eog_var": eog_var
    }
    
    eog_df = pd.DataFrame.from_dict(data)
    
    return eog_df

''' PERCLOS '''
def get_perclos_labels(client):
    mat_data = read_mat("data/SEED-VIG/perclos_labels/" + client)

    perclos = mat_data["perclos"]
    boundary = min(perclos) + (max(perclos)-min(perclos))*0.25
    
    # TRUE=Somnoliento; False=Despierto
    y = perclos >= boundary
    
    data = {
        "y_reg": perclos,
        "y_class": y
    }
    
    perclos_df = pd.DataFrame.from_dict(data)
    
    return perclos_df

''' FINAL AGGREGATION '''
def get_client_data(client):
    psd = get_psd_data(client)
    
    eog = get_eog_data(client)
    
    perclos = get_perclos_labels(client)
    
    client_data = psd.join(eog.join(perclos))
    return client_data

def client_remove_psd_outliers(client_data):
    z = stats.zscore(client_data[['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']])
    client_data_o = client_data[(z < 3).all(axis=1)]
    
    return client_data_o


''' Ejecución del script '''
def main():
    for f in listdir("./data/SEED-VIG/Raw_Data/"):
        m = re.search("(.{1,2})_.*", f)
        if m:
            num_client = int(m.group(1))
            client = f

            client_data = get_client_data(client)
            client_data_o = client_remove_psd_outliers(client_data)

            dest_filename = f"./data/horizontal/empresa_{(num_client%2)+1}/cliente_{num_client}.csv"

            # Si existe ya el fichero => sujeto con dos experimentos => se 'stackean' los dataframes para generar uno mas grande
            if path.exists(dest_filename):
                existing_client_data = pd.read_csv(dest_filename)

                client_data_o = pd.concat([existing_client_data, client_data_o])

            client_data_o.to_csv(f"./data/horizontal/empresa_{(num_client%2)+1}/cliente_{num_client}.csv", index=False, header=True)
            
if __name__ == "__main__":
    main()