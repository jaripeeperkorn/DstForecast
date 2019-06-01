# DstForecast
LSTM neural network to forecast Dst index. Master Thesis Jari Peeperkorn, KU Leuven, 2018-2019.

Inputdata set 1 uses input features Dst, |B|, Bz, V, n and T
Inputdata set 2 uses input features Dst, Kp, AE,|B|, Bz

Folder data contains:
- Datafixer: script to fill gaps in data by linear interpolation
- Datacorr: script to make correlation analysis between parameters used in thesis

Folder Inputdata 1 contains:
- Hyperarameter code
- Training code
- Different saved models (3,4,5 an 6 hours forecast)

Folder Inputdata 2 contains:
- Hyperarameter code
- Training code
- Different saved models (3,4,5 an 6 hours forecast)
