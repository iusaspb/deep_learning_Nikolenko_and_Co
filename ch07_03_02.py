from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    from gensim.corpora.wikicorpus import WikiCorpus
    from gensim.models.callbacks import CallbackAny2Vec
    from gensim.models.phrases import Phrases, Phraser, original_scorer
    #
    #  загрузить с https://dumps.wikimedia.org/ruwikisource/20190701/
    #  https://dumps.wikimedia.org/ruwiki/20190720/ruwiki-20190720-pages-articles-multistream.xml.bz2
    #
    # wiki = WikiCorpus(r"D:\tmp\ruwikisource-20190701-pages-articles-multistream.xml.bz2",dictionary=False)
    wiki = WikiCorpus(r"D:\tmp\ruwiki-20190720-pages-articles-multistream.xml.bz2",dictionary=False)
    bigram = Phrases(wiki.get_texts())
    bigram_transformer = Phraser(bigram)
    bigram.save('ruwiki.bigram')
    bigram_transformer.save('ruwiki.bigram_transformer')
    def text_generator_bigram():
        for text in wiki.get_texts():
            yield bigram_transformer[ [ word for word in text ] ]
    def text_generator_trigram():
        for text in wiki.get_texts():
            yield trigram_transformer[ bigram_transformer[[ word for word in text ] ] ]
    trigram = Phrases(text_generator_bigram())
    trigram_transformer = Phraser(trigram)

    trigram.save('ruwiki.trigram')
    trigram_transformer.save('ruwiki.trigram_transformer')


    from gensim.models.word2vec import Word2Vec
    epochs=100

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


    model = Word2Vec(size=300, window=7, min_count=trigram.min_count, workers=10, callbacks=[SimpleLogger(epochs)])
    model.build_vocab( text_generator_trigram() )
    model.train( text_generator_trigram(),total_words=len(trigram.vocab),epochs=epochs)
    model.save("ruwiki_300.model")
    with open('Таблица 7.4. Примеры ближайших соседей в модели word2vec.txt', 'w') as  fout:
        def print2(text):
            print(text)
            fout.write(text)
            fout.write('\n')
        def show_similar(word):
            try:
                for s in model.most_similar(word):
                    print2(str(s))
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
