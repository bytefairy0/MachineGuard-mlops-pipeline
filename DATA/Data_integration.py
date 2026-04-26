import pandas as pd

# 1. Load Data
ai4i = pd.read_csv('DATA/ai4i2020.csv')
nasa_cols = ['unit_id', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
nasa = pd.read_csv('DATA/train_FD001.txt', sep=r'\s+', header=None, names=nasa_cols)

# 2. NASA -> Calculate RUL
# Vectorized calculation is faster than a merge
nasa['RUL'] = nasa.groupby('unit_id')['cycle'].transform('max') - nasa['cycle']

# 3. Map NASA -> AI4I schema
nasa_mapped = nasa[['s2', 's3', 's8', 's11', 'cycle', 'RUL']].copy()
nasa_mapped.columns = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]',
    'Tool wear [min]', 'RUL'
]
nasa_mapped['Source'] = 'NASA'

# 4. AI4I -> Calculate RUL (FIXED & FAST)
# Changed 'Machine ID' to 'Product ID'
fail_indices = ai4i.index[ai4i['Machine failure'] == 1]

def fast_rul(idx):
    future = fail_indices[fail_indices >= idx]
    return (future[0] - idx) if not future.empty else 300

ai4i['RUL'] = ai4i.index.map(fast_rul)

# 5. Keep ONLY matching columns
cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'RUL']
ai4i_mapped = ai4i[cols].copy()
ai4i_mapped['Source'] = 'AI4I'

# 6. Merge
unified_data = pd.concat([ai4i_mapped, nasa_mapped], ignore_index=True)
unified_data.to_csv('MachineGuard_Unified.csv', index=False)

print("✅ Unified dataset ready (AI4I and NASA merged successfully)")