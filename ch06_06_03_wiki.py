import os
import numpy as np
from keras.models import Model
from keras.layers import concatenate
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Input

START_CHAR = '\b'
END_CHAR = '\t'
PADDING_CHAR = '\a'
chars = set( [START_CHAR, '\n', END_CHAR] )
output_fname='сhar_samplers_skip_layer_connection.txt'
input_fname='data/onegin.txt'
model_fname='ch06_06_3_layers_wiki'
output_fname='data/сhar_samplers_3_layers.txt'
batchoutput_fname='logs/'
batch_size = 16
output_dim=128
num_epochs=6#1000
#  составить список всех символов в тексте и построить индексы
with open(input_fname, 'r', encoding='utf_8') as f:
    for line in f:
        chars.update( list(line.strip().lower()) )
char_indices = { c : i for i,c in enumerate(sorted(list(chars))) }
char_indices[PADDING_CHAR] = 0
indices_to_chars = { i : c for c,i in char_indices.items() }
num_chars = len(chars)
def get_one(i, sz):
    res = np.zeros(sz)
    res[i] = 1
    return res
# так как char_indices[PADDING_CHAR] = 0, то в случае c == PADDING_CHAR ничего не делаем
char_vectors = { c  : ( np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars)) for c,v  in char_indices.items()}
# разбиваем текст на преддложения.
# предложения оканчиваются  .!?  ТОЛЬКО В КОНЦЕ СТРОКИ
sentence_end_markers = set( '.!?' )
sentences = []
current_sentence = ''
with open( input_fname, 'r', encoding='utf_8') as f:
    for line in f:
        s = line.strip().lower()
        if len(s) > 0:
            current_sentence += s + '\n'

        if len(s) == 0 or s[-1] in sentence_end_markers:  # -1 - предложение не может оканчиваться в середине строки.
            current_sentence = current_sentence.strip()
            if len(current_sentence) > 10: # выкидываем короткие предложения
                sentences.append(current_sentence)
                current_sentence = ''

def get_matrices(sentences):
    """Набор предлождений преобразуем во вход и выход модели.
       размерность входных данных
       кол-во предложений x (max длина предложения +1) x  вектор длины num_chars с одной единицей
       кроме случая PADDING_CHAR, в котором все эдементы вектора 0

       выход - сдвинутый на единицу вход
       так как мы хотим предсказывать следующий символ по предыдущему
    """
    # +1 для START_CHAR или END_CHAR. ЭТОГО НЕ БЫЛО В КНИГЕ
    max_sentence_len = np.max([ len(x) for x in sentences ])+1
    X = np.zeros((len(sentences), max_sentence_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), max_sentence_len, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        char_seq = (START_CHAR + sentence + END_CHAR).ljust(max_sentence_len+1, PADDING_CHAR)
        for t in range(max_sentence_len):
            X[i, t, :] = char_vectors[char_seq[t]]
            y[i, t, :] = char_vectors[char_seq[t+1]]
    return X,y


# подготовить datasets
# тестовые данные - 5% от всех
test_indices = np.random.choice(range(len(sentences)), int(len(sentences) * 0.05))
sentences_test = [ sentences[x] for x in test_indices]
X_test, y_test = get_matrices(sentences_test)
# обучающие данные
sentences_train = [ sentences[x] for x in set(range(len(sentences))) - set(test_indices) ]
sentences_train = sorted(sentences_train, key = lambda x : len(x))

def generate_batch():
    """Генерируем очередной batch """
    while True:
        for i in range( int(len(sentences_train) / batch_size) ):
            sentences_batch = sentences_train[ i*batch_size : (i+1)*batch_size ]
            yield get_matrices(sentences_batch)



class CharSampler(Callback):
    """Генератор образца """
    def __init__(self, char_vectors, model):
        self.char_vectors = char_vectors
        self.model = model

    def on_train_begin(self, logs={}):
        self.epoch = 0
        if os.path.isfile(output_fname):
            os.remove(output_fname)

    def sample( self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def sample_one(self, T):
        result = START_CHAR
        while len(result)<500:
            Xsampled = np.zeros( (1, len(result), num_chars) )
            for t,c in enumerate( list( result ) ):
                Xsampled[0,t,:] = self.char_vectors[ c ]
            ysampled = self.model.predict( Xsampled, batch_size=1 )[0,:]
            yv = ysampled[len(result)-1,:]
            selected_char = indices_to_chars[ self.sample( yv, T ) ]
            if selected_char==END_CHAR:
                break
            result = result + selected_char
        return result


    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch + 1
        if self.epoch % 50 == 0:
            print("\nEpoch %d text sampling:" % self.epoch)
            with open(output_fname, 'a' ,encoding='utf_8') as outf:
                outf.write('\n===== Epoch %d =====\n' % self.epoch)
                for T in [0.3, 0.5, 0.7, 0.9, 1.1]:
                    print('\tsampling, T = %.1f...' % T)
                    for _ in range(5):
                        self.model.reset_states()
                        res = self.sample_one(T)
                        outf.write('\nT = %.1f\n%s\n' % (T, res[1:])) # [1:] - откидывем START_CHAR


vec = Input(shape=(None, num_chars))
l1 = LSTM(output_dim=output_dim, activation='tanh', return_sequences=True)(vec)
l1_d = Dropout(0.2)(l1)
input2 = concatenate([vec, l1_d])
l2 = LSTM(output_dim=output_dim, activation='tanh', return_sequences=True)(input2)
l2_d = Dropout(0.2)(l2)
input3 = concatenate([vec, l2_d])
l3 = LSTM(output_dim=output_dim, activation='tanh', return_sequences=True)(input3)
l3_d = Dropout(0.2)(l3)
input_d = concatenate([l1_d, l2_d, l3_d])
dense3 = TimeDistributed(Dense(output_dim=num_chars))(input_d)
output_res = Activation('softmax')(dense3)
model = Model(input=vec, output=output_res)
model.compile(loss='categorical_crossentropy',optimizer=Adam(clipnorm=1.), metrics=['accuracy'])
def size_LSTM(input_size, output_size):
    # 4 - candidate cell state+input gate+forget gate+ output gate
    return 4*(input_size*output_size + input_size*output_size + output_size)

num_params = size_LSTM(num_chars, output_dim) + size_LSTM(num_chars + output_dim, output_dim) + size_LSTM(num_chars + output_dim, output_dim)
num_params += num_chars*3*output_dim + 3*output_dim # dense
print(f'количество параметров в моделе = {num_params}')


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        if (batch+1) % 100 == 0:
            with open( batchoutput_fname, 'a' ,encoding='utf_8') as outf:
                for i in range(100):
                    outf.write('%d\t%.6f\t%.6f\n' % (batch+i-99, self.loss[i-100], self.acc[i-100]))


cb_sampler = CharSampler(char_vectors, model)
cb_logger = CSVLogger('logs/' + model_fname + '.log')
cb_checkpoint= ModelCheckpoint('checkpoints')

hist = model.fit_generator(
    generate_batch(),
    int(len(sentences_train) / batch_size) * batch_size,nb_epoch=num_epochs,
    verbose=2,
    validation_data = (X_test, y_test)
    ,callbacks=[
        # cb_logger,
        cb_sampler
        # , cb_checkpoint
                ]
)

import matplotlib.pyplot as plt
t = np.arange(num_epochs)
fig, axs = plt.subplots(2, 1)
fig.canvas.set_window_title('Рис. 6.15. Графики функции потерь и точности на тренировочном и валидационном множествах модель с тремя слоями LSTM, «Евгений Онегин»')
axs[0].plot(t, hist.history['loss'], label='Train')
axs[0].plot(t, hist.history['val_loss'], label='Test')
axs[0].set_xlabel('эпохи')
axs[0].set_ylabel('Функция потерь')
axs[0].set_title('Функция потерь')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t, hist.history['acc'], label='Train')
axs[1].plot(t, hist.history['val_acc'], label='Test')
axs[1].set_xlabel('эпохи')
axs[1].set_ylabel('Точность')
axs[1].set_title('Точность')

axs[1].grid(True)
axs[1].legend()

fig.tight_layout()
plt.show()
