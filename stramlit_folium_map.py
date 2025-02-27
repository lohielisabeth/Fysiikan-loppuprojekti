import streamlit as st
import pandas as pd 
import folium 
from streamlit_folium import st_folium

path1 = "Location.csv"
df = pd.read_csv(path1)

path2 = "Linear Accelerometer.csv"
df2 = pd.read_csv(path2)

st.title('Data of the walk')

from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, fs, nyq, order): 
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, nyq, order): 
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

T = df2['Time (s)'][len(df2['Time (s)'])-1] - df2['Time (s)'][0]
n = len(df2['Time (s)'])
fs = n/T
nyq = fs/2
order = 3
cutoff = 1/(0.2)

df2['filter_a_z'] = butter_lowpass_filter( df2['Z (m/s^2)'], cutoff, fs, nyq, order)

from scipy.signal import find_peaks

peaks, _ = find_peaks(df2['filter_a_z'], distance=fs*0.2) 

valleys, _ = find_peaks(-df2['filter_a_z'], distance=fs*0.2)
total_steps = min(len(peaks), len(valleys))

st.write("Step count calculated using filtering :",total_steps, "steps")