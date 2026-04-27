from sympy import evaluate

import thinkdsp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

#signal
cos_sig = thinkdsp.CosSignal(freq=440,amp=1.0,offset=0)
sin_sig = thinkdsp.SinSignal(freq=880,amp=0.5,offset=0)
mix = cos_sig+sin_sig

#makewave
wave = mix.make_wave(duration=0.5, start=0, framerate=11025)
# wave.plot()
# plt.show()

#segment
period = mix.period
segment = wave.segment(start=0, duration=3*period)
segment.plot()
plt.show()

#read the wave
bird_wave = thinkdsp.read_wave(r'/\Acoustic\Animals-audio-dataset\bird\0a7c2a8d_nohash_0.wav')
wave.write(filename='simple_wave.wav')
# thinkdsp.play_wave(filename='simple_wave.wav',player='mp3')
bird_wave.plot()
plt.show()
spect = bird_wave.make_spectrum()
spect.plot()
plt.show()
audio = Audio(data=bird_wave.ys,rate=bird_wave.framerate)

#Spectrums
spectrum = wave.make_spectrum()
spectrum.plot()
plt.show()
spectrum.low_pass(cutoff=600, factor=0.01)
spectrum.plot()
plt.show()
# wave = spectrum.make_wave()
# segment = wave.segment(start=0, duration=3*period)
# segment.plot()
# plt.show()

#wave object
wave.ys *= 2
wave.ts += 1
# wave.scale(2)
# wave.shift(1)

#Signal Object
# class Sinusoid(thinkdsp.Signal):
#     def __init__(self, freq=440, amp=1.0, offset=0,func=np.sin):
#         thinkdsp.Signal.__init__(self)
#         self.freq = freq
#         self.amp = amp
#         self.offset = offset
#         self.func = func
#     def evaluate(self, ts):
#         phase = 2*np.pi*self.freq*ts+self.offset
#         ys = self.amp*self.func(phase)
#         return ys
#     def make_wave(self, duration=1, start=0, framerate=11025):
#         n = round(duration*framerate)
#         ts = start+np.arange(n)/framerate
#         ys = self.evaluate(ts)
#         return thinkdsp.Wave(ys,ts,framerate)
