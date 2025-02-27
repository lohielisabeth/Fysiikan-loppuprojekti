import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

path1 = "Location.csv"
df = pd.read_csv(path1)

path2 = "Linear Accelerometer.csv"
df2 = pd.read_csv(path2)

st.title('The data of my walk')

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
fs = n / T
nyq = fs / 2
order = 3
cutoff = 1 / 0.2

df2['filter_a_y'] = butter_lowpass_filter(df2['Y (m/s^2)'], cutoff, fs, nyq, order)

peaks, _ = find_peaks(df2['filter_a_y'], distance=fs * 0.2) 
valleys, _ = find_peaks(-df2['filter_a_y'], distance=fs * 0.2)
total_steps = min(len(peaks), len(valleys))

st.write("Step count by filtering:", total_steps, "steps")

time = df2.iloc[:, 0].values
signal = df2.iloc[:, 1].values
dt = time[1] - time[0]
N = len(signal)  
fourier = fft(signal, N) 
psd = (fourier * np.conj(fourier)) / N  
freq = fftfreq(N, dt)

step_freq_range = (1, 3) 
step_freqs = np.logical_and(freq > step_freq_range[0], freq < step_freq_range[1])

step_psd = psd[step_freqs]
step_count = len(step_psd) 

st.write("Step count estimated based on Fourier analysis:", step_count, "steps")

R = 6371000.0  

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c 
    return distance

total_distance = 0
for i in range(1, len(df)):
    lat1, lon1 = df.iloc[i-1]['Latitude (°)'], df.iloc[i-1]['Longitude (°)']
    lat2, lon2 = df.iloc[i]['Latitude (°)'], df.iloc[i]['Longitude (°)']
    total_distance += haversine(lat1, lon1, lat2, lon2)

total_distance_rounded = round(total_distance, 1)
st.write("Total distance: ", total_distance_rounded, 'm')

total_time = df['Time (s)'].iloc[-1] - df['Time (s)'].iloc[0]  
total_minutes = int(total_time // 60)
total_seconds = int(total_time % 60)

st.write("Total time:", total_minutes, "min", total_seconds, "s")

average_speed = total_distance / total_time  
average_speed_rounded = round(average_speed, 2)
st.write("Average speed: ", average_speed_rounded, "m/s")

step_length = total_distance / total_steps 
step_length_rounded = round(step_length, 2)
st.write("Step length:", step_length_rounded, "m")

st.title('Filtered acceleration data (Y-component)')
df2['Time (min)'] = df2['Time (s)'] / 60

chart_data = pd.DataFrame({
    'Time (min)': df2['Time (min)'],
    'filter_a_y': df2['filter_a_y']
})

st.line_chart(chart_data.set_index('Time (min)'))

st.title('Filtered acceleration data (Y-component) between 0.4s and 2s')
df2['Time (min)'] = df2['Time (s)'] / 60

filtered_chart_data = df2[(df2['Time (min)'] >= 0.4) & (df2['Time (min)'] <= 2)]
st.line_chart(filtered_chart_data.set_index('Time (min)')[['filter_a_y']])

st.title('Power Spectrum')

def plot_power_spectrum(signal, fs):
    N = len(signal)
    fourier = fft(signal)
    freqs = fftfreq(N, 1/fs)
    psd = np.abs(fourier)**2 / N
    pos_freqs = freqs[:N//2]
    pos_psd = psd[:N//2]
    
    chart_data = pd.DataFrame({
        'freq': pos_freqs,
        'psd': pos_psd
    })
    
    st.line_chart(chart_data.set_index('freq'))
    
    return chart_data

if 'filter_a_y' in df2.columns:
    dt = df2['Time (s)'].iloc[1] - df2['Time (s)'].iloc[0]  
    fs = 1 / dt  
    chart_data = plot_power_spectrum(df2['filter_a_y'], fs)
else:
    st.error("The DataFrame does not contain the 'filter_a_y' column.")

st.title('Route on the map')
start_lat = df['Latitude (°)'].mean()
start_long = df['Longitude (°)'].mean()

map = folium.Map(location=[start_lat, start_long], zoom_start=16, scrollWheelZoom=False, dragging=False) 
folium.PolyLine(df[['Latitude (°)', 'Longitude (°)']], color='deeppink', weight=2.5, opacity=1).add_to(map)
st_map = st_folium(map, width=900, height=650)
