from gensim.models import Word2Vec
from multiprocessing import Queue
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast
import torch
import jieba
import numpy as np

# nigo
def sim(text1, text2):
    words = list(jieba.cut(text2))
    text2 = max(words, key=len)
    seg_text1 = list(jieba.cut(text1))
    seg_text2 = list(jieba.cut(text2))
    model = Word2Vec(
        [seg_text1, seg_text2], vector_size=100, window=5, min_count=1, sg=0
    )
    model.train([seg_text1, seg_text2], total_examples=2, epochs=10)
    vec1 = np.mean([model.wv[word] for word in seg_text1 if word in model.wv], axis=0)
    vec2 = np.mean([model.wv[word] for word in seg_text2 if word in model.wv], axis=0)
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


def similarity(q: Queue, start, end, title_list, text):
    max_prob = float('-inf')
    ret = 0
    for i in tqdm(range(start, end)):
        prob = sim(title_list[i][0], text)
        if prob > max_prob:
            ret = i
            max_prob = prob
    q.put((ret, max_prob))
