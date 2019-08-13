#Рис. 6.16. Графики функции потерь и точности на тренировочном множестве при обучении моделей с одним и тремя слоями LSTM на русскоязычной «Википедии»
import numpy as np
import matplotlib.pyplot as plt
log_fname1 = 'logs/ch06_06_03.log'
log_fname3 = 'logs/ch06_06_04.log'
history1={'loss':[],'acc':[]}
history3={'loss':[],'acc':[]}
with open( log_fname1, 'r' ,encoding='utf_8') as log_1, open( log_fname3, 'r' ,encoding='utf_8') as log_3 :
    for line in log_1:
        fields = line.split()
        history1['loss'].append(float(fields[1]))
        history1['acc'].append(float(fields[2]))
    for line in log_3:
        fields = line.split()
        history3['loss'].append(float(fields[1]))
        history3['acc'].append(float(fields[2]))

    len = min(len(history1['loss']),len(history3['loss']))
    t = np.arange(len)
    fig, axs = plt.subplots(2, 1)
    fig.canvas.set_window_title('Рис. 6.16. Графики функции потерь и точности на тренировочном множестве при обучении моделей с одним и тремя слоями LSTM на русскоязычной «Википедии»')
    axs[0].plot(t, history1['loss'][:len], label='1 слой')
    axs[0].plot(t, history3['loss'][:len], label='3 слоя')
    axs[0].set_xlabel('эпохи')
    axs[0].set_ylabel('Функция потерь')
    axs[0].set_title('Функция потерь')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, history1['acc'][:len], label='1 слой')
    axs[1].plot(t, history3['acc'][:len], label='3 слоя')
    axs[1].set_xlabel('эпохи')
    axs[1].set_ylabel('Точность')
    axs[1].set_title('Точность')

    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()
    plt.show()


