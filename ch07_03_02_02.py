import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class SimpleLogger(CallbackAny2Vec):
    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch = 0
        self.batch = 0

    def on_train_begin(self, model):
        print("Train starts...")

    def on_epoch_begin(self, model):
        self.batch = 0
        print(f"Epoch {self.epoch}({self.epochs}) start")

    def on_batch_begin(self, model):
        print(f"Batch {self.batch} start")

    def on_batch_end(self, model):
        print(f"Batch {self.batch} end")
        self.batch += 1

    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch} end")
        self.epoch += 1

    def on_train_end(self, model):
        print("Train finished")

model = Word2Vec.load("ruwiki_300.model")
# открыть файл без 'with' если используете с python -i ch07_03_02_02.py,
# чтобы поиграить с моделью после выполнения скрипта
# не забудьте только потом выполнить fout.close()
fout = open('Таблица 7.4. Примеры ближайших соседей в модели word2vec.txt', 'w')
# with open('Таблица 7.4. Примеры ближайших соседей в модели word2vec.txt', 'w') as  fout:
def print2(text):
    print(text)
    fout.write(text)
    fout.write('\n')
def show_similar(word):
    print2(str(word))
    try:
        for s in model.most_similar(word):
            print2(str(s))
        print2('')
    except KeyError:
        print2('No '+word)

show_similar('машина_времени')
show_similar('машинное_обучение')
show_similar('микеланджело')
show_similar('микеланджело_антониони')
show_similar('вторая_мировая_война')
show_similar('великая_отечественная_война')
show_similar('александр_македонский')
show_similar('леонид_ильич_брежнев')
