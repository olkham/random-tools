import librosa
import librosa.core
# import librosa.display
import numpy as np
import os

from enum import Enum
import pyaudio
from typing import Any
from threading import Thread
import time

class AudioSource():

    class Source(Enum):
        File = 1
        Microphone = 2

    def __init__(self) -> None:
        self.running = False
        self.available_devices = [Any]
        self.data = [float]
        self.buffer = [float]
        self.progress = 0
        self.elapsed = 0
        self.start_time = 0
        self.ready = False
        self.new_data = False

        self.isopen = False

        self.realtime_playback = True

        self.SAMPLE_RATE = 44100 #sample rate
        self.FORMAT = pyaudio.paInt16 #conversion format for PyAudio stream
        self.CHANNELS = 1 #microphone audio channels
        self.SAMPLE_SIZE = 1024 #number of samples to take per read
        self.SAMPLE_LENGTH = int(self.SAMPLE_SIZE*1000/self.SAMPLE_RATE) #length of each sample in ms
        self.DURATION = 1.0
        self.BUFFER_LENGTH = 1

    def configure(self, sample_rate, channels, num_samples):
        self.SAMPLE_RATE = sample_rate
        self.CHANNELS = channels
        self.SAMPLE_SIZE = num_samples

    def set_sample_duration(self, duration):
        self.DURATION = duration
        self.SAMPLE_SIZE = int(self.SAMPLE_RATE * self.DURATION)
        self.SAMPLE_LENGTH = int(self.SAMPLE_SIZE*1000/self.SAMPLE_RATE) #length of each sample in ms

    def set_buffer_length(self, samples):
        self.BUFFER_LENGTH = samples


    def set_sample_step(self, step):
        self.STEP = step

    def list_available_devices(self):
        pa = pyaudio.PyAudio()
        info = pa.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                device = pa.get_device_info_by_host_api_device_index(0, i)
                self.available_devices.append(device)
                print("Input Device id ", i, " - ", device.get('name'))

    def open(self, src):
        if os.path.isfile(src):
            self.signal_source = self.Source.File
            self.src = src
            # self.data, rate = librosa.load(self.src,
            #                                 sr=self.SAMPLE_RATE, 
            #                                 offset=0, 
            #                                 duration=self.DURATION)
        
        if type(src) is int:
            self.signal_source = self.Source.Microphone
            self.src = src
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format = self.FORMAT,
                                        channels = self.CHANNELS,
                                        input_device_index=self.src,
                                        rate = self.SAMPLE_RATE,
                                        input = True,
                                        frames_per_buffer = self.SAMPLE_SIZE)

        self.isopen = True

    def got_new_data(self):
        self.ready = True
        self.new_data = True
        self.elapsed = time.time() - self.start_time

    def update(self):
        if self.signal_source == self.Source.File:
            while self.running:

                if self.realtime_playback:
                    offset = self.elapsed
                else:
                    offset = self.progress

                self.data, rate = librosa.load(self.src,
                                            sr=self.SAMPLE_RATE, 
                                            offset=offset, 
                                            duration=self.DURATION)
                self.got_new_data()
                self.progress += self.DURATION
                if len(self.data) == 0:
                    self.stop()
                
        if self.signal_source == self.Source.Microphone:
            while self.running:
                input_data = self.stream.read(self.SAMPLE_SIZE, exception_on_overflow = False)
                self.data = np.frombuffer(input_data, np.int16) / (2**16)

                if self.BUFFER_LENGTH > 1:
                    if len(self.buffer) == 1:
                        self.buffer = self.data
                    else:
                        self.buffer = np.concatenate([self.buffer,self.data])
                        if len(self.buffer) >= len(self.data)*self.BUFFER_LENGTH+1:
                            self.buffer = self.buffer[len(self.data):]
                else:
                    self.buffer = self.data

                self.got_new_data()

    def play(self):
        if self.running:
            return

        self.start_time = time.time()
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.isopen = False

    def read_next_frame(self, normalise=False):
        if self.signal_source == self.Source.File:
            self.data, rate = librosa.load(self.src,
                                            sr=self.SAMPLE_RATE, 
                                            offset=self.progress, 
                                            duration=self.DURATION)
            self.progress += self.DURATION

            if len(self.data) == 0:
                self.stop()
                return self.data

            if normalise:
                return self.norm(self.data)
            return self.data

    def read(self, normalise=False, step=False):
        if self.signal_source == self.Source.File:
            self.data, rate = librosa.load(self.src,
                                            sr=self.SAMPLE_RATE, 
                                            offset=self.progress, 
                                            duration=self.DURATION)

            if step:
                self.step_forward()

            if len(self.data) == 0:
                self.stop()
                return self.data

            if normalise:
                return self.norm(self.data)
            return self.data

    def pause(self):
        pass

    def step_forward(self, step=None):
        if self.signal_source == self.Source.File:
            if step is None:
                self.progress += self.STEP
            else:
                self.progress += step

    def step_backward(self, step=None):
        if self.signal_source == self.Source.File:
            if step is None:
                self.progress -= self.STEP
            else:
                self.progress -= step

    def rewind(self, step):
        pass

    def fastforward(self, step):
        pass

    def norm(self, a):
        if len(a) > 0:
            return np.interp(a,[np.min(a),np.max(a)],[-1,1]).astype('float32')
        return a

    def get_latest_samples(self, normalise=False):

        if not self.running:
            self.play()

        while not self.ready or not self.new_data:
            pass

        self.new_data = False
        if normalise:
            return self.norm(self.buffer)
        return self.buffer

    def get_spectrum(self, n_mels=256, fft_size=4096, normalise=True, fmin=0, fmax=None):
        S = librosa.feature.melspectrogram(y=self.get_latest_samples(normalise=normalise),
                                        sr=self.SAMPLE_RATE, 
                                        n_mels=n_mels, 
                                        fmin=fmin,
                                        fmax=fmax,
                                        n_fft=fft_size)

            # convert the slices to amplitude
        Xdb = librosa.amplitude_to_db(S, ref=np.max, top_db=80.0)

        # convert image to colormap
        image_gr = np.interp(Xdb, [-80, 0], [0, 255]).astype('uint8')
        return image_gr

def example():
    import cv2
    audio_source = AudioSource()
    audio_source.list_available_devices()

    audio_source.set_sample_duration(0.01)
    audio_source.set_buffer_length(200)
    # audio_source.set_sample_step(0.1)
    audio_source.open(1)

    cv2.namedWindow("Result",0)
    key = ''
    while key != ord('q'):
        #y = audio_source.get_latest_samples(normalise=True)
        image_gr = audio_source.get_spectrum(n_mels=512, normalise=False)
        image_gr = cv2.flip(image_gr, 0)
        image = cv2.applyColorMap(image_gr, 20)
        cv2.imshow("Result", image)
        key = cv2.waitKey(1)

if __name__ == '__main__':
    example()