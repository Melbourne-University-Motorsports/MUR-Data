import sys, pathlib
import pandas as pd
proj_src = pathlib.Path(__file__).resolve().parents[1] / "motec-to-csv" / "src"
sys.path.append(str(proj_src))
from motec_converter import parse_race_data, to_pandas
sys.path.remove(str(proj_src))
import matplotlib.pyplot as plt
import numpy as np


def normalize(x):
    return (x - np.mean(x)) / np.std(x)



base_dir = pathlib.Path("dyno_stuff")
ld_files = sorted(base_dir.glob("*.ld"))


df_dyno = pd.read_csv(base_dir / "dyno.csv", index_col=0, parse_dates=True)

dyno_rpm = df_dyno['Motor Speed (rpm)']

dyno_rpm = normalize(dyno_rpm)


runs = {}

for f in ld_files:
    race_data = parse_race_data(str(f))
    segments = to_pandas(race_data)
    df = pd.concat(segments).sort_index()

    motec_rpm = df['Car.Data.Motor.MotorRPM'].dropna().values
    
    if len(motec_rpm) < 100:
        continue
    
    motec_rpm = normalize(motec_rpm)

    # create normalized x-axis (0 → 1)
    x_dyno = np.linspace(0, 1, len(dyno_rpm))
    x_motec = np.linspace(0, 1, len(motec_rpm))

    plt.figure()
    plt.plot(x_dyno, dyno_rpm, label="dyno")
    plt.plot(x_motec, motec_rpm, label=f"motec: {f.name}")
    plt.legend()
    plt.title(f.name)
    plt.show()


#global_df = pd.concat(all_segments, ignore_index=False).sort_index()
#print(global_df.columns)


#Sl #10889 20260313 140852.1d  


f= 'dyno_stuff/S1_#10889_20260313_140852.ld'

race_data = parse_race_data(str(f))
segments = to_pandas(race_data)
df = pd.concat(segments).sort_index()

motec_rpm = df['Car.Data.Motor.MotorRPM'].dropna().values

# normalize
d = normalize(dyno_rpm)
m = normalize(motec_rpm)

# resample to same length (important!)
m_resampled = np.interp(
    np.linspace(0, 1, len(d)),
    np.linspace(0, 1, len(m)),
    m
)

# correlation
score = np.corrcoef(d, m_resampled)[0, 1]

print(f"Correlation score: {score:.4f}")
