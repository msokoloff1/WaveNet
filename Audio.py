import scipy.io.wavfile as wave
import numpy as np





def getSample(source = "./Samples/island_music_x.wav"):
    loadedAudio = wave.read(source)
    sampleRate = loadedAudio[0]
    audio = loadedAudio[1]
    #Ensure that values come in between 0 and 255 (Otherwise we have to perform quantization)
    #assert(loadedAudio[1].dtype == 'uint8')
    return [audio, sampleRate]
    
    
    


def writeSample(filename, data):
    rate = getSample()[1]
    wave.write(filename, rate, data.astype('uint8'))
    
    



def quantToWave(output):
        casted = output.astype('float32')
        signal = 2 * (casted / 255) - 1
        magnitude = (1 / 255) * ((256)**np.abs(signal) - 1)
        return np.sign(signal) * magnitude


