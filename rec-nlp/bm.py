import en_core_web_sm
import pandas as pd
from gensim.summarization.bm25 import BM25
import numpy as np
import pickle


data = pd.read_csv(r'data\all_2.csv')

nlp = en_core_web_sm.load(disable=['ner', 'tagger', 'parser'])

cleaner = lambda x: [str(a).lower() for a in nlp(x)
                     if not (a.is_punct or not a.is_alpha)]

title = data['title'] + ' ' + data['abstract']

title = title.apply(cleaner).tolist()

bm25 = BM25(title)

res = bm25.get_scores(['text', 'classification'])

with open('bm25_bow.pkl', 'wb') as output:
    pickle.dump(bm25, output, pickle.HIGHEST_PROTOCOL)

with open('../bm25.pkl', 'rb') as input:
    bm = pickle.load(input)

res_2 = bm.get_scores(['text', 'classification'])