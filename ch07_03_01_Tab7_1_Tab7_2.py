import re
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.phrases import Phrases, Phraser, original_scorer
from collections import defaultdict
#
#  загрузить с https://dumps.wikimedia.org/ruwikisource/20190701/
#
wiki = WikiCorpus(r"D:\tmp\ruwikisource-20190701-pages-articles-multistream.xml.bz2",dictionary=False)
bigram = Phrases(wiki.get_texts())
bigram_transformer = Phraser(bigram)

def text_generator_bigram():
    for text in wiki.get_texts():
        yield bigram_transformer[ [ word for word in text ] ]

vocab = bigram.vocab
bi_vocab = defaultdict(list)
for p,c in vocab.items():
    if re.findall('_', p.decode('utf-8')):
        bi_vocab[c].append(p)
len_vocab = float(len(vocab))
min_count = float(bigram.min_count)
corpus_word_count = float(bigram.corpus_word_count)
print('Таблица 7.1. Примеры биграмм из русскоязычной «Википедии»')
for c in sorted(bi_vocab.keys(),reverse=True)[:20]:
    for val in bi_vocab[c]:
        [worda, wordb] = re.split(b'_', val)
        score = original_scorer(
            worda_count=float(vocab[worda]),
            wordb_count=float(vocab[wordb]),
            bigram_count=float(c),
            len_vocab=len_vocab, min_count=min_count, corpus_word_count= corpus_word_count)
        print(val.decode('utf-8'), c, score)

trigram = Phrases(text_generator_bigram())
vocab = trigram.vocab
tri_vocab = defaultdict(list)
for p,c in vocab.items():
    if len(re.findall(b'_', p)) ==2:
        tri_vocab[c].append(p)

len_vocab = float(len(vocab))
min_count = float(trigram.min_count)
corpus_word_count = float(trigram.corpus_word_count)
print('Таблица 7.2. Примеры триграмм из русскоязычной «Википедии»')
for c in sorted(tri_vocab.keys(),reverse=True)[:20]:
    for val in tri_vocab[c]:
        [worda,wordb] = re.split(b'_', val,1)
        score = original_scorer(
            worda_count=float(vocab[worda]),
            wordb_count=float(vocab[wordb]),
            bigram_count=float(c),
            len_vocab=len_vocab, min_count=min_count, corpus_word_count= corpus_word_count)
        print(val.decode('utf-8'), c, score)

bigram.score_item('машина','времени')