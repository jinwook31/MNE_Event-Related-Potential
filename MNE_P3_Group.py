import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from time import time

import mne
from mne.preprocessing import ICA, create_eog_epochs
from mne.channels import combine_channels

pList = ['P01', 'P02', 'P04', 'P05']
ytick = 9

def analysis(partNum):
    fPath = '../Raw_data/' + partNum + '.vhdr'

    #Read raw data with MNE
    raw = mne.io.read_raw_brainvision(fPath, preload=False)
    raw.pick_types(meg=False, eeg=True, eog=False).load_data()
    raw.drop_channels(['ExG 1','ExG 2', 'A2'])  # Remove unused channel


    #Get event code from annotations
    events, events_id = mne.events_from_annotations(raw, event_id='auto')
    raw, events = raw.resample(256, events=events)


    # Sensor Location (Topological)
    montage = mne.channels.make_standard_montage(kind='standard_1020')  # 10-20 system montage
    raw.set_montage(montage)

    # Signal Processing
    raw.filter(l_freq=0.5,h_freq=None,fir_design='firwin')   #high pass filter with fir filter
    raw.filter(l_freq=None,h_freq=40,fir_design='firwin')    #low pass filter  with fir filter
    raw.set_eeg_reference(ref_channels='average')


    # Event discription
    events_id={'Normal0':10002,'Normal1':10003,'Normal2':10004,'Normal3':10005,'odd0':10007,'odd1':10008,'odd2':10009,'odd3':10010}  #redifine the events_id
    events = mne.pick_events(events, include=[10002, 10003, 10004, 10005,  10007, 10008, 10009, 10010])  #pick events  that we interested


    # ICA
    epochs_ica = mne.Epochs(raw, events=events, event_id=events_id, preload=True, tmin=-0.8, tmax=1.0)

    from autoreject import get_rejection_threshold
    reject = get_rejection_threshold(epochs_ica)

    ica=ICA(n_components=28, method='fastica', random_state=89).fit(epochs_ica, reject=reject)  #define the parameter of ica and fit it to epochs

    ica.apply(raw)  # remove the component that we selected


    # Epoch (with Rejection) & Average
    epochs = mne.Epochs(raw, events=events, event_id=events_id, baseline=(-0.2, 0), preload=True, tmin=-0.2, tmax=0.5, reject=dict(eeg=150e-6))
    
    # Exclude Participant depending on the rejection
    if epochs.drop_log_stats() > 25:
        print("P" + partNum + " was excluded! (over 25)")
        #return
    #else:
    epochs.apply_baseline(baseline=(-0.2, 0))   
    epochs.save('./MNE Result/' + partNum + '.fif', overwrite=True)

    for i in range(0, 4):
        normalCond = 'Normal' + str(i)
        oddCond = 'odd' + str(i)
        evoked_normal = epochs[normalCond].average()
        evoked_odd = epochs[oddCond].average()

        # Plot ERP Graph
        condition_index={'odd':evoked_odd, 'normal':evoked_normal}
        plot = mne.viz.plot_compare_evokeds(condition_index, picks=['Pz'], ci=None, combine='mean',legend='lower right',show_sensors='upper left', show=False, ylim=dict(eeg=[-ytick, ytick])) #plot compared ERP
        plot[0].savefig('./MNE Result/Condition ' + str(i) + '_' + partNum + '.jpg')

        # Plot Difference Waveform Graph
        evokeds_diff = mne.combine_evoked([evoked_odd, evoked_normal], weights=[1, -1])
        plot = mne.viz.plot_compare_evokeds({'Mismatch-Match':evokeds_diff}, picks=['Pz'], show_sensors='upper right', combine='mean', ylim=dict(eeg=[-ytick, ytick]), title='Difference Wave (Condition_' + str(i) + ')')
        plot[0].savefig('./MNE Result/Differnece waveform_Condition ' + str(i) + '_' + partNum + '.jpg')



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
    for i in range(0, 4):
        normalCond = 'Normal' + str(i)
        oddCond = 'odd' + str(i)
        evoked_normal = epochs[normalCond].average()
        evoked_odd = epochs[oddCond].average()

        # Plot ERP Graph
        condition_index={'odd':evoked_odd, 'normal':evoked_normal}
        plot = mne.viz.plot_compare_evokeds(condition_index, picks=['Pz'], ci=None, combine='mean',legend='lower right',show_sensors='upper left', show=False, ylim=dict(eeg=[-ytick, ytick])) #plot compared ERP
        plot[0].savefig('./MNE Result/Condition ' + str(i) + ' Grand Average.jpg')

        # Plot Difference Waveform Graph
        evokeds_diff = mne.combine_evoked([evoked_odd, evoked_normal], weights=[1, -1])
        plot = mne.viz.plot_compare_evokeds({'Mismatch-Match':evokeds_diff}, picks=['Pz'], show_sensors='upper right', combine='mean', ylim=dict(eeg=[-ytick, ytick]), title='Difference Wave (Condition_' + str(i) + ')')
        plot[0].savefig('./MNE Result/Differnece waveform_Condition ' + str(i) + '.jpg')





def main():
    for p in pList:
        analysis(p)
        pass

    grandAvg()


if __name__ == "__main__":
    main()

