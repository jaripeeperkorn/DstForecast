# DstForecast
LSTM neural network to forecast Dst index. Master Thesis Jari Peeperkorn, KU Leuven, 2018-2019.
Also presented as a poster on ESWW 2019 https://register-as.oma.be/esww16/contributions/public/S13-P1/S13-P1-03-PeeperkornJari/ESWW_Poster.pdf


Inputdata set 1 uses input features Dst, |B|, Bz, V, n and T
Inputdata set 2 uses input features Dst, Kp, AE,|B|, Bz

Folder data contains:
- Datafixer: script to fill gaps in data by linear interpolation
- Datacorr: script to make correlation analysis between parameters used in thesis

Folder Inputdata 1 contains:
- Hyperarameter code
- Training code
- Different saved models (3,4,5 an 6 hours forecast)
- f1 notebook that can be used to calculate recall, precision and f1 score

Folder Inputdata 2 contains:
- Hyperarameter code
- Training code
- Different saved models (3,4,5 an 6 hours forecast)
- - f1 notebook that can be used to calculate recall, precision and f1 score
