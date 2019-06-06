from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot
from keras.models import load_model
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os, sys, cv2, librosa, shutil


class MyFirstGUI(QDialog):

    def __init__(self):
        super().__init__()
        loadUi('visualizer.ui', self)

        self.waveGraph.save_done_signal.connect(self.analyze)

        pixmap = QPixmap('resources/record.ico')
        ButtonIcon = QIcon(pixmap)
        self.recordButton.setIcon(ButtonIcon)
        self.model = None
        self.modelBox.setCurrentIndex(0)
        # 強迫執行所有events
        self.numberLabel.repaint()
        QApplication.instance().processEvents()



    # https://stackoverflow.com/questions/14311578/event-signal-is-emmitted-twice-every-time
    @pyqtSlot()
    def on_modelButton_clicked(self):
        pre = ''
        if self.enButton.isChecked():
            pre = 'EN_'
        elif self.chButton.isChecked():
            pre = 'CH_'

        self.numberLabel.setText('Loading Model ' + pre + self.modelBox.currentText() + '.h5 ......')
        self.numberLabel.setStyleSheet('QLabel { background-color : WhiteSmoke;  color : DarkCyan; }')
        # force qApp to process repaint event
        # 不知為何只repaint label時好時壞，要repaint整個dialog
        self.numberLabel.repaint()
        self.repaint()
        QApplication.instance().processEvents()

        self.model = load_model('models/' + pre + self.modelBox.currentText() +'.h5')
        self.numberLabel.setText('Loading Complete!')
        self.numberLabel.setStyleSheet('QLabel { background-color : WhiteSmoke;  color : DarkCyan; }')
        # self.numberLabel.repaint()
        # QApplication.instance().processEvents()

    def on_recordButton_toggled(self, status):

        if self.model == None:
            self.numberLabel.setText('Select Your Desire Model First!')
            self.numberLabel.setStyleSheet('QLabel { background-color : WhiteSmoke;  color : red; }')
            # self.numberLabel.repaint()
            # QApplication.instance().processEvents()
            self.recordButton.setChecked(False)
            return

        if status: # start record
            try:
                shutil.rmtree('output')
            except:
                pass
            os.makedirs('output', mode=0o777)
            self.recordButton.setText('Stop')
            self.recordButton.setStyleSheet('QPushButton{ background-color: LightCoral;border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px;}')

            self.waveGraph.frames = []
            self.waveGraph.finishrecord = False
            self.waveGraph.stream_setup()
        else:
            self.recordButton.setText('Record')
            self.recordButton.setStyleSheet('QPushButton{ background-color: LightSkyBlue ;border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px;}')
            # callback裡面叫paComplete來stop_stream
            self.waveGraph.finishrecord = True

    def analyze(self):
        numbers = ''
        sound = AudioSegment.from_wav('output/user_input.wav')
        dBFS = sound.dBFS
        chunks = split_on_silence(sound,
                                  min_silence_len=250,  # mine:250
                                  silence_thresh=dBFS - 15,
                                  keep_silence=100  # optional
                                  )
        # self.progressBar.setMaximum(len(chunks))

        for i, chunk in enumerate(chunks):
            chunk.export(f'output/{i}.wav', bitrate="192k", format="wav")

            y, sr = librosa.load(f'output/{i}.wav')

            result = None
            if self.modelBox.currentText() == 'CQT_CNN':
                CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr, n_bins=84)), ref=np.max)
                CQT = cv2.resize(CQT.astype('float'), (60, 84), interpolation=cv2.INTER_CUBIC)
                if self.enButton.isChecked():
                    CQT = (CQT - (-87.7)) / (4.55 - (-87.7))
                elif self.chButton.isChecked():
                    CQT = (CQT - (-86.9)) / (6.64 - (-86.9))
                result = self.model.predict(CQT.reshape(1, 84, 60, 1))

            elif self.modelBox.currentText() == 'STFT_CNN':
                STFT = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                STFT = cv2.resize(STFT.astype('float'), (50, 1025), interpolation=cv2.INTER_CUBIC)
                STFT = STFT[:300, :]

                if self.enButton.isChecked():
                    STFT = (STFT - (-88.4)) / (5.71 - (-88.4))
                elif self.chButton.isChecked():
                    STFT = (STFT - (-86.4)) / (2.52 - (-86.4))
                result = self.model.predict(STFT.reshape(1, 300, 50, 1))

            elif self.modelBox.currentText() == 'MFCC_RNN':
                if self.enButton.isChecked():
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)
                    mfcc = cv2.resize(mfcc.astype('float'), (30, 25), interpolation=cv2.INTER_LINEAR)
                    mfcc = (mfcc - (-705.)) / (305. - (-705.))
                    result = self.model.predict(mfcc.reshape(1, 25, 30))
                elif self.chButton.isChecked():
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc = cv2.resize(mfcc.astype('float'), (30, 13), interpolation=cv2.INTER_LINEAR)
                    mfcc = (mfcc - (-688.)) / (294. - (-688.))
                    result = self.model.predict(mfcc.reshape(1, 13, 30))

            l = np.squeeze(result).argsort()
            numbers = numbers + str(l[-1]) + ', '

        self.numberLabel.setText(f'"{len(chunks)}" Spoken-Digits Detected : '+numbers[:-2])
        self.numberLabel.setStyleSheet('QLabel { background-color : WhiteSmoke;  color : blue; }')


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = MyFirstGUI()
    window.setWindowTitle('Spoken-Digit Recognizer')
    window.show()
    sys.exit(app.exec_())
