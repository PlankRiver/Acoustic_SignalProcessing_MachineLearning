import thinkdsp
import numpy as np
import matplotlib.pyplot as plt

#Triangle Signal
signal = thinkdsp.TriangleSignal(200)
wave = signal.make_wave(duration=0.5,framerate=10000)
wave.plot()
plt.show()
#spectrum
spectrum = wave.make_spectrum()
spectrum.plot()
plt.show()

#Square Signal
signal = thinkdsp.SquareSignal(200)
wave = signal.make_wave(duration=0.5,framerate=10000)
wave.plot()
plt.show()
#spectrum
spectrum = wave.make_spectrum()
spectrum.plot()
plt.show()