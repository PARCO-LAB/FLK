from scipy.signal import butter, filtfilt

class LPF:
    def __init__(self,latency,order,cutoff,fs) :
        self.latency = latency
        self.b, self.a = butter(N=order, Wn=cutoff, btype='low', analog=False, fs=fs)

    