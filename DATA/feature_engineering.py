import pandas as pd
import numpy as np

# 1. LOAD DATA
df = pd.read_csv('MachineGuard_Unified.csv')

# 2. LAYER 1: RAW SENSORS (With Simulated Measurement Noise)
# We add 0.5% random noise to simulate real-world sensor jitter. 
# This prevents the model from "perfectly" memorizing specific values.
def add_noise(series, level=0.005):
    return series * (1 + np.random.normal(0, level, len(series)))

df['Air_Temperature'] = add_noise(df['Air temperature [K]'])
df['Process_Temp'] = add_noise(df['Process temperature [K]'])
df['Rotational_Speed'] = add_noise(df['Rotational speed [rpm]'])
df['Torque'] = add_noise(df['Torque [Nm]'])
df['Life_Consumed'] = df['Tool wear [min]']

# 3. LAYER 2: STRESS PROXY (Same logic, but it now consumes noisy data)
def envelope_scale(s, low, high): return (s - low) / (high - low)
n_torque = envelope_scale(df['Torque'], 0, 100)
n_temp = envelope_scale(df['Process_Temp'], 280, 350)
n_speed_inv = 1 - envelope_scale(df['Rotational_Speed'], 1000, 3000)
df['Stress_Index'] = ((n_torque * 0.4) + (n_temp * 0.4) + (n_speed_inv * 0.2)).clip(0, 1)

# 4. LAYER 3: STOCHASTIC LABELING (Fixing the "Pure Rule" issue)
# We add a "Probabilistic Flip." 5% of the time, the label is determined 
# by the sensors even if the RUL says otherwise. This simulates "Random Failures."
is_end_of_life = (df['RUL'] <= 15)
is_high_strain = (df['Stress_Index'] > 0.75)
random_failure = (np.random.random(len(df)) < 0.02) # 2% chance of sudden failure

df['Is_Failure'] = (is_end_of_life | (is_high_strain & (df['RUL'] <= 40)) | random_failure).astype(int)

# 5. HEALTH TAXONOMY
conditions = [(df['Is_Failure'] == 1), (df['RUL'] <= 70) | (df['Stress_Index'] > 0.65)]
df['Health_Status'] = np.select(conditions, ['Critical', 'Warning'], default='Healthy')

# --- FINAL FREEZE ---
df_final = df[['Source', 'Air_Temperature', 'Process_Temp', 'Rotational_Speed', 
               'Torque', 'Life_Consumed', 'Stress_Index', 'RUL', 
               'Is_Failure', 'Health_Status']]

df_final.to_csv('MachineGuard_v1_2_Generalizable.csv', index=False)
print("✅ v1.2 Generalizable Build: Measurement noise and sudden failure simulation added.")