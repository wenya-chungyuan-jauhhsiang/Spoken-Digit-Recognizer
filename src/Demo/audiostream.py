import numpy as np
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg
import struct
import pyaudio
from scipy.fftpack import fft
import wave


class AudioStream(pg.GraphicsLayoutWidget):

    save_done_signal = pyqtSignal()

    def __init__(self, parent):
        super().__init__()

        # antialias=True的話可以做一些圖形平滑化，但會降低一些performance
        pg.setConfigOptions(antialias=False)

        self.record = False
        self.finishrecord = False
        self.frames = []
        self.test = None
        self.traces = dict()

        self.waveform = self.addPlot( title='WAVEFORM', row=1, col=1, )
        self.spectrum = self.addPlot( title='SPECTRUM', row=2, col=1, )

        # PyAudio Stuff
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2

        # waveform and spectrum x points
        self.x1 = np.arange(0, 2 * self.CHUNK, 2)
        self.x2 = np.linspace(0, self.RATE / 2, int(self.CHUNK / 2))

        self.traces['waveform'] = self.waveform.plot(pen='c', width=3)
        self.waveform.setYRange(0, 255, padding=0)
        self.waveform.setXRange(0, 2 * self.CHUNK, padding=0.005)
        self.traces['spectrum'] = self.spectrum.plot(pen='m', width=3)

        self.spectrum.setLogMode(x=True, y=False)
        self.spectrum.setYRange(0, 1, padding=0)
        self.spectrum.setXRange(np.log10(20), np.log10(self.RATE / 2), padding=0.005)

    def stream_setup(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            # macos下output要設False，don't know why
            output=False,
            frames_per_buffer=self.CHUNK,
            # 用callback法就不會有buffer堵塞問題
            stream_callback=self.update,
            # open會自動start_stream，這邊取消掉，比較好控制
            start=False
        )
        self.stream.start_stream()

    # callback會在不同thread做
    def update(self, in_data, frame_count, time_info, status):
        wf_data = in_data
        wf_data = struct.unpack(str(2 * self.CHUNK) + 'B', wf_data)
        wf_data = np.array(wf_data, dtype='b')[::2] + 128
        self.traces['waveform'].setData(self.x1, wf_data)

        sp_data = fft(np.array(wf_data, dtype='int8') - 128)
        sp_data = np.abs(sp_data[0:int(self.CHUNK / 2)]) * 2 / (128 * self.CHUNK)
        self.traces['spectrum'].setData(self.x2, sp_data)

        # 本來就只有在錄的時候才會進來，就不要用self.record了
        self.frames.append(in_data)
        if self.finishrecord == True:
            self.save_wav()
            return (in_data, pyaudio.paComplete)

        return (in_data, pyaudio.paContinue)

    # 不讓使用者因為操控圖形導致display error
    def mouseMoveEvent(self, ev):
        ev.ignore()
    def mouseReleaseEvent(self, ev):
        ev.ignore()
    def mousePressEvent(self, ev):
        ev.ignore()
    def wheelEvent(self, ev):
        ev.ignore()

    def save_wav(self):
        waveFile = wave.open('output/user_input.wav', 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(self.p.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()
        self.save_done_signal.emit()
