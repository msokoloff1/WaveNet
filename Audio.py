import scipy.io.wavfile as wave







def getSample(source = "./Samples/island_music_x.wav"):
    loadedAudio = wave.read(source)
    sampleRate = loadedAudio[0]
    audio = loadedAudio[1]
    #Ensure that values come in between 0 and 255 (Otherwise we have to perform quantization)
    assert(loadedAudio[1].dtype == 'uint8')
    return [audio, sampleRate]
    
    
    


