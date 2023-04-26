import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from time import time

import mne
from mne.preprocessing import ICA, create_eog_epochs
from mne.channels import combine_channels


def analysis(partNum):
    fPath = './' + partNum + '/' + partNum + '.set'

    #Read raw data with MNE
    raw = mne.io.read_raw_eeglab(fPath, preload=False)
    raw.pick_types(meg=False, eeg=True, eog=False).load_data()
    raw.drop_channels(['ExG 1','ExG 2', 'A2','Packet Counter', 'TRIGGER'])  # Remove unused channel


    #Get event code from annotations
    events, events_id = mne.events_from_annotations(raw, event_id='auto')


    # Sensor Location (Topological)
    montage = mne.channels.make_standard_montage(kind='standard_1020')  # 10-20 system montage
    raw.set_montage(montage)

    # Signal Processing
    raw.filter(l_freq=0.5,h_freq=None,fir_design='firwin')   #high pass filter with fir filter
    raw.filter(l_freq=None,h_freq=40,fir_design='firwin')    #low pass filter  with fir filter
    raw.set_eeg_reference(ref_channels='average')


    # Event discription
    events_id={'odd':1,'response':2,'dummy':3,'normal':4}  #redifine the events_id
    events = mne.pick_events(events, include=[1, 2, 3, 4])  #pick events  that we interested


    # ICA
    ica=ICA(n_components=29, method='fastica', random_state=89).fit(raw)  #define the parameter of ica and fit it to epochs
    ica.apply(raw)  # remove the component that we selected


    # Epoch (with Rejection) & Average
    epochs = mne.Epochs(raw, events=events, event_id=events_id, baseline=(-0.2, 0), preload=True, tmin=-0.2, tmax=0.5, reject=dict(eeg=150e-6))
    
    # Exclude Participant depending on the rejection
    if epochs.drop_log_stats() > 25:
        print("P" + partNum + " was excluded! (over 25)")
        return
    else:
        epochs.apply_baseline(baseline=(-0.2, 0))   
        epochs.save('./MNE Result/' + partNum + '.fif', overwrite=True)
        evoked_normal = epochs['normal'].average()
        evoked_odd = epochs['odd'].average()

    # Plot Graph
    condition_index={'odd':evoked_odd,'normal':evoked_normal}
    plot = mne.viz.plot_compare_evokeds(condition_index, picks=['Pz'], ci=None, combine='mean',legend='lower right',show_sensors='upper left', show=False) #plot compared ERP
    plot[0].savefig('./MNE Result/' + partNum + '.jpg')



# Plot Grand Average (P3)
def grandAvg():
    fPath = './MNE Result/'
    fifFolder = [f for f in os.listdir(fPath) if f.endswith('.fif')]

    # Read each participant Epochs
    res = []
    for fif in fifFolder:
        epochs = mne.read_epochs(fPath + str(fif))
        res.append(epochs)
    
    epochs = mne.concatenate_epochs(res)  # Concate and make full epochs

    # Plot Grand Average
    evoked_normal = epochs['normal'].average()
    evoked_odd = epochs['odd'].average()
    condition_index={'odd':evoked_odd,'normal':evoked_normal} 
    plot = mne.viz.plot_compare_evokeds(condition_index, picks=['Pz'], ci=None, combine='mean',legend='lower right',show_sensors='upper left', show=False) #plot compared ERP
    plot[0].savefig('./MNE Result/Grand Average.jpg')



def main():
    for i in range(1, 10):
        #analysis(str(i))
        pass

    grandAvg()


if __name__ == "__main__":
    main()

