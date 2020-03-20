<p align="center">
  <a href=#>
    <img src="resources/microphone.png" alt="Spoken-Digit Recognizer" width="96" height="96">
  </a>
  <h2 align="center">語音數字辨識專案 (Spoken-Digit Recognizer)</h2>
  <div align="center">
    本project旨在運用Keras建立Model，辨識使用者說的中/英數字，並使用GUI呈現。
  </div>
</p>

<p align="center"><a href="https://github.com/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/tree/master/src/split_audio"><img src="resources/Split--Audio-src-brightgreen.svg" alt="" /></a>
<a href="https://github.com/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/tree/master/src/Spectrogram_CNN"><img src="resources/Spectrogram+CNN-src-brightgreen.svg" alt="" /></a>
<a href="https://github.com/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/tree/master/src/MFCC_RNN"><img src="resources/MFCC+RNN-src-brightgreen.svg" alt="" /></a>
<a href="https://github.com/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/tree/master/src/Gan"><img src="resources/GAN-src-brightgreen.svg" alt="" /></a>
<a href="https://github.com/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/tree/master/src/Demo"><img src="resources/Demo-src-brightgreen.svg" alt="" /></a></p>

<p align="center"><a href="https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/split_audio.ipynb"><img src="resources/Split--Audio-doc-blue.svg" alt="" /></a>
<a href="https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/Spectrogram_CNN_doc.ipynb"><img src="resources/Spectrogram%2BCNN-doc-blue.svg" alt="" /></a>
<a href="https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/MFCC_RNN_doc.ipynb"><img src="resources/MFCC%2BRNN-doc-blue.svg" alt="" /></a>
<a href="https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/GAN_doc.ipynb"><img src="resources/GAN-doc-blue.svg" alt="" /></a>
<a href="https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/Demo_doc.ipynb"><img src="resources/Demo-doc-blue.svg" alt="" /></a></p>

## 先看段Demo吧！
### [Spoken-Digit Recognizer – Demo (YouTube)](https://www.youtube.com/watch?v=_yKum1DxPJM)
<p align="center">
<a href=https://www.youtube.com/watch?v=_yKum1DxPJM>
    <img src="resources/demo.gif" alt="demo" width="720" height="450">
</a>
</p>

Demo詳細介紹請點連結
- ### [(Detailed) Demo](https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/Demo_doc.ipynb)

## Dataset
**所有使用data皆置於[dataset.rar](https://github.com/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/dataset.rar)**
### 英文
- [pannous on github](https://github.com/pannous/tensorflow-speech-recognition?fbclid=IwAR1tThhKhbMM_BnKE4SK16qcbuGdw1gJw7iWVVyEhDk9vZFF5Z8E6rjuWUs)
    - 連結內spoken_numbers_pcm.tar含2400筆.wav檔，為15位不同人唸英文數字(0~9)的單數字音檔(160/人)
- 3位contributer每人自錄160筆，與上述相加共2880筆

### 中文
- 3位contributer每人自錄500筆，共1500筆
    - 每筆data為中文數字數字0~9單數字音檔，每人一個數字錄50筆

## 目標
1. 使用者對麥克風説一串中/英文數字(0~9)，程式能辨識使用者說了哪些數字  
2. 使用生成對抗網路GAN來生成音檔，即讓程式產出數字0~9的音檔

## Implementation
對目標1，先將包含多數字的音檔分割，再使用不同種model來辨識，詳細介紹請點連結
- ### [(Detailed) Split Audio](https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/split_audio.ipynb)
- ### [(Detailed) Spectrogram + CNN](https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/Spectrogram_CNN_doc.ipynb)    
- ### [(Detailed) MFCC + RNN](https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/MFCC_RNN_doc.ipynb)

對目標2，使用inverse-STFT方式，詳細介紹請點連結
- ### [(Detailed) GAN](https://nbviewer.jupyter.org/github/wenya-chungyuan-jauhhsiang/Spoken-Digit-Recognizer/blob/master/docs/GAN_doc.ipynb)

## 問題討論
### 1. 聲紋影響
    - 一個沒有經過我們model訓練過的人聲，若進行辨識測驗時的平均正確率會較低
    - 我們認為這和聲紋相關，也就是同樣的字由不同人發聲的訊號頻譜存在差異
### 2. 中/英文
    - 我們初期是以英文數字為輸入音訊，後期則發現英文其實在發音上相較中文有更多的變化性，如某些子音的發音屬於清音，會較容易被誤判為靜音
    - 英文對於發音並沒有制式的音調規則，例如有些字會因語氣不同而音調不同，這導致我們model的辨識正確率並不理想
    - 後來我們選擇嘗試中文，由於中文絕大多數發音是濁音，且抑揚頓挫已有明確定義，因此訓練出的model辨識正確率果然如我們預期，有明顯的提升

## Contributors
[WenYa Lin](https://github.com/wenyalintw)、[ChungYuan Hsu](https://github.com/ChungYuanHsu)、[JauhHsiang Lan](https://github.com/r07522749)
