import numpy
import pydub
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
pydub.AudioSegment.converter = r"C:\libav\bin\ffmpeg.exe"  # set path of ffmpeg.exe

# for i in range(1):                                         # numbers from 0 to 9     
#     sound = AudioSegment.from_wav(f'emily_wav\{i}.wav')
#     dBFS = sound.dBFS
#     chunks = split_on_silence(sound, 
#             min_silence_len = 250,                         # minimum length of silence:250 ms 
#             silence_thresh = dBFS-16,                      # threshhold to divide voice and silence
#             keep_silence = 100                             # time left before and after each voice cut:100 ms
#         )
#     os.mkdir(f'emily_wav\{i}_Emily')
#     for j, chunk in enumerate(chunks):
#         chunk.export(f'emily_wav\{i}_Emily\{i}_Emily_{j}.wav',bitrate = "192k",format = "wav")

                                        # numbers from 0 to 9     
sound = AudioSegment.from_wav(f'3486790251.wav')
dBFS = sound.dBFS
chunks = split_on_silence(sound, 
        min_silence_len = 250,                         # minimum length of silence:250 ms 
        silence_thresh = dBFS-16,                      # threshhold to divide voice and silence
        keep_silence = 100                             # time left before and after each voice cut:100 ms
    )
for j, chunk in enumerate(chunks):
    chunk.export(f'split_audio_{j}.wav',bitrate = "192k",format = "wav")
