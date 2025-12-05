import os
import pandas as pd
import matplotlib.pyplot as plt

# ─── 1. Paths & read ─────────────────────────────────────────────────────────
base_dir   = 'postProcessing'
sub_dir    = 'minmax'
case_dir   = '0'
filename   = 'fieldMinMax.dat'
data_dir   = os.path.join(base_dir, sub_dir, case_dir)
dat_path   = os.path.join(data_dir, filename)
csv_path   = os.path.join(data_dir, 'fieldMinMax.csv')

df_minmax = pd.read_table(
    dat_path,
    sep='\t',
    header=None,
    skiprows=[0,1],
    names=[
        'time',
        'field',
        'min',
        'location_min',
        'processor_min',
        'max',
        'location_max',
        'processor_max'
    ]
)

base_dir   = 'postProcessing'
sub_dir    = 'average'
case_dir   = '0'
filename   = 'volFieldValue.dat'
data_dir   = os.path.join(base_dir, sub_dir, case_dir)
dat_path   = os.path.join(data_dir, filename)
csv_path   = os.path.join(data_dir, 'fieldMinMax.csv')
pdf_path   = os.path.join(base_dir, 'plots.pdf')

df_mean = pd.read_table(
    dat_path,
    sep='\t',
    header=None,
    skiprows=[0,1,2,3],
    names=[
        'time',
        'angle',
        'skewness',
        'volume'
    ]
)

# ─── 2. Clean ─────────────────────────────────────────────────
df_minmax['field'] = df_minmax['field'].str.strip()

# ─── 3. Export cleaned CSV (no index) ────────────────────────────────────────
df_minmax.to_csv(csv_path, index=False)

# ─── 4. Select the two fields ───────────────────────────────────────────────
df_skew   = df_minmax[df_minmax['field'] == 'skewness'].reset_index(drop=True)
df_skew.insert(0,'mean', df_mean['skewness'])
df_nonortho = df_minmax[df_minmax['field'] == 'nonOrthoAngle'].reset_index(drop=True)
df_nonortho.insert(0,'mean', df_mean['angle'])
df_volume = df_minmax[df_minmax['field'] == 'cellVolume'].reset_index(drop=True)
df_volume.insert(0,'mean', df_mean['volume'])


# ─── 5. Plot setup ─────────────────────────────────────────────────────────
plt.rc('font', size=18)
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)

# common plotting function
def plot_field(ax, data, name):
    # ax.plot(data['time'], data['min'],  marker='o', linestyle='-', label=f'{name} min')
    ax.fill_between(data['time'], data['min'], data['max'], alpha=0.4)
    ax.plot(data['time'], data['mean'],marker='o', linestyle='-')
    negatives = data[data['min']<0]
    ax.scatter(negatives['time'], negatives['min'],marker='o', c='r',zorder=10)
    # ax.plot(data['time'], data['max'],  marker='o', linestyle=':', label=f'{name} max')
    ax.set_title(name)
    ax.set_xlabel('Time')
    ax.set_ylabel(name)
    ax.grid(True)
    # ax.legend()

# ─── 6. Draw the two panels ─────────────────────────────────────────────────
plot_field(axes[0], df_skew,       'skewness')
plot_field(axes[1], df_nonortho, 'nonOrthoAngle')
plot_field(axes[2], df_volume, 'cellVolume')

fig.tight_layout()

# ─── 7. Save to PDF ─────────────────────────────────────────────────────────
fig.savefig(pdf_path, format='pdf')
print(f"Saved plots to {pdf_path}")
