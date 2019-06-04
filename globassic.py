import numpy as np
from scipy import signal
from IPython import display
import matplotlib.pyplot as plt

"""
Collection of basic audio functions
"""

def playSound(sound_arr, rate = 41400):
    display.display( display.Audio(sound_arr, rate=rate) )
    
def plotSpectrogram(sourceArr, sampleRate, expo=0.2):
    freqs, times, spectrogram = signal.spectrogram(sourceArr, sampleRate)
    plt.pcolormesh(times, freqs, spectrogram**(expo))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()    
    
def __segment_frames(input_sound, dft_size, hop_size):
    num_frames = int((len(input_sound) / (hop_size)) + 0.5)
    input_sound = np.append(input_sound, np.zeros( dft_size ))
    
    frames = []
    for i in range(num_frames):
        slice_start = i * hop_size
        slice_end = slice_start + dft_size
        frames.append(input_sound[slice_start:slice_end])
    return np.array(frames)

from numpy.fft import rfft
def __stft_forward(input_sound, dft_size, hop_size, zero_pad, window):
    if window is not None:
        assert len(window) == dft_size
    frames = __segment_frames(input_sound, dft_size, hop_size)
    if window is not None:
        frames = np.multiply(frames, window)
    fft_frames = np.array( [rfft(frames[i], n=len(frames[i]) + zero_pad) for i in range(len(frames))] )
    matrix = np.stack(fft_frames, axis=1)
    return matrix

from numpy.fft import irfft
def __stft_inverse(input_sound, dft_size, hop_size, zero_pad, window):
    if window is not None:
        assert len(window) == dft_size
    spectrums = np.transpose(input_sound)
    snippets = np.array( [irfft(spectrums[i]) for i in range(len(spectrums))] )
    snippets = np.array( [snippets[i][:] for i in range(len(snippets))] )  
    if window is not None:
        snippets = np.multiply(snippets, window)
    output = np.zeros(len(snippets) * hop_size + dft_size, dtype='complex')
    for i in range(len(snippets)):
        for j in range(len(snippets[i])):
            output[i*hop_size + j] += snippets[i][j]
    return output


def stft(input_sound, dft_size, hop_size, zero_pad=0, window=None):
    if len(np.shape(input_sound)) == 1:
        return __stft_forward(input_sound, dft_size, hop_size, zero_pad, window)
    elif len(np.shape(input_sound)) == 2:
        return __stft_inverse(input_sound, dft_size, hop_size, zero_pad, window)