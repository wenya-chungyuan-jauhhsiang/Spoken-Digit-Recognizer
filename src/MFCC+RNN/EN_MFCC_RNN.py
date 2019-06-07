import librosa
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras import optimizers


#%% 把spoken_numbers_pcm內的.wav檔做MFCC，並把矩陣存入mfcc_npy (直接存矩陣不存圖，就沒有resolution的問題)

def store_mfcc_npy(fname):
    y, sr = librosa.load('spoken_numbers_pcm/'+fname)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=25)

    # 如欲輸出包含MFCC一二階微分值時使用，然經測試效果不佳
    # mfcc_delta = librosa.feature.delta(mfcc,mode='nearest')
    # mfcc_delta2 = librosa.feature.delta(mfcc, order=2,mode='nearest')
    # mfcc_final = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

    mfcc_final = cv2.resize(mfcc.astype('float'), (30, 25), interpolation=cv2.INTER_LINEAR)
    np.save('mfcc_npy/'+fname, mfcc_final)

filename = glob.glob('spoken_numbers_pcm/**.wav')

for i in range(len(filename)):
    # mac
    #filename[i] = filename[i].split('/')[1]
    # window
    filename[i] = filename[i].split('\\')[1]
    store_mfcc_npy(filename[i])


#%% 讀取MFCC並設定為訓練資料與測試資料

x_list = []
y_list = []
t_list = []
filename_npy = glob.glob('mfcc_npy/**.npy')

for i in range(len(filename_npy)):
    # mac
    #t_list.append([filename_npy[i].split('/')[1].split('_')[1],np.load(filename_npy[i]),filename_npy[i].split('/')[1][0]])
    # window
    t_list.append([filename_npy[i].split('\\')[1].split('_')[1],np.load(filename_npy[i]),filename_npy[i].split('\\')[1][0]])

# 將資料按人員排列
# def takeFirst(elem):
#     return elem[0]
# t_list.sort(key = takeFirst)

# 將資料打亂
random.shuffle(t_list)

# 將資料設定為訓練資料與測試資料
for i in range(len(filename_npy)):
    x_list.append(t_list[i][1])
    y_list.append(t_list[i][2])
x_train = np.asarray(x_list[:2500])
y_train = np.asarray(y_list[:2500])
x_test = np.asarray(x_list[2500:])
y_test = np.asarray(y_list[2500:])

# normalized和1-hot
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


#%% 建立RNN模型並訓練

model = Sequential()
model.add(LSTM(units=128,return_sequences=True,recurrent_initializer='orthogonal',input_shape=(25, 30)))
model.add(Dropout(0.3))
model.add(LSTM(units=128,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=128))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
print(model.summary())
models = model.fit(x_train,y_train,batch_size=128,epochs=1500,verbose=2,shuffle = True,validation_data = (x_test, y_test))


#%% 訓練結果呈現

fig = plt.figure(figsize=(16,5))
plt.plot(models.history["acc"],label='train')
plt.plot(models.history["val_acc"],label='test')
plt.title("model training accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0))
plt.show()


#%% score

score = model.evaluate(x_test, y_test, batch_size=10000)
print("Loss: %f" %score[0])
print("testing accuracy: %f" %(score[1]*100))


#%% 儲存model

model.save('MFCC_RNN.h5')