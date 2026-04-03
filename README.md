# MLMAMidtermProject


### 1. Data Preprocessing 
#### Dataset Overview 
- Wearable sensor data collected from 15 nurses to predict stress levels. 
- High-frequency physiological signals: 
    - Heart Rate (HR): beats per minute 
    - Electrodermal Activity (EDA): skin conductance 
    - Skin Temperature (TEMP) 
    - Accelerometer (X, Y, Z) 
    - Datetime: timestamp of each reading, including year, day, and exact time 
    - Label: self-reported stress level (0=no stress, 1=medium, 2=high)

- Each nurse data spans from a minumum of 1 to maximum of 13 days and consists of time-stamped observations. 

#### Preprocessing Pipeline 
- We processed data individually for each nurse to avoid data leakage and account for individual physiological baselines, especially given that we only have 15 nurses only. 
- The X, Y, Z accelerometer values are converted into magnitude using basic distance formula, to get `acc_mag`. 
- The four physiological signal features we focus on are: 
    - `HR`, `EDA`, `TEMP`, `acc_mag` or Acceleration Magnitude 
- Features normalized per nurse using z-score normalization 
- For each participant/nurse: 
    - Time window aggregation: raw time-series aggregated into fixed 60-second windows using `df_nurse.resample('60s')`. This creates fixed, disjoint time window bins: 
    ```
    08:00:00–08:00:59  → window 1  
    08:01:00–08:01:59  → window 2  
    08:02:00–08:02:59  → window 3 
    ```
    Aggregating data points within that window frame into exactly one window. (we can maybe consider using overlapping if we want smoother predictions? but tbh we have a pretty large dataset )
- Feature engineering 

