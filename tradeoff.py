from gensim.models import Word2Vec
import jieba
import numpy as np
from tqdm import trange, tqdm
import tqdm
import gensim
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import scipy.stats as stats
import time
from opencc import OpenCC

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_b = AutoModel.from_pretrained(model_name)


# Jieba + Word2Vec
def sim_w(text1, text2):
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


# Jieba + FastText
def sim_f(text1, text2):
    seg_text1 = list(jieba.cut(text1))
    seg_text2 = list(jieba.cut(text2))

    # Train FastText model
    model = gensim.models.FastText(
        [seg_text1, seg_text2], vector_size=100, window=5, min_count=1, sg=0
    )

    vec1 = np.mean([model.wv[word] for word in seg_text1 if word in model.wv], axis=0)
    vec2 = np.mean([model.wv[word] for word in seg_text2 if word in model.wv], axis=0)

    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


# Hugging Face Transformer Bert
def sim_b(text1, text2):
    inputs = tokenizer(
        [text1, text2], return_tensors="pt", padding=True, truncation=True
    )

    with torch.no_grad():
        outputs = model_b(**inputs)

    embeddings1 = torch.from_numpy(outputs.last_hidden_state[0].mean(dim=0).numpy())
    embeddings2 = torch.from_numpy(outputs.last_hidden_state[1].mean(dim=0).numpy())

    similarity = (embeddings1.dot(embeddings2)) / (
        torch.norm(embeddings1) * torch.norm(embeddings2)
    )
    return similarity.item()


def score(wordSimStd, wordSimPre):
    corrCoef = np.corrcoef(wordSimStd, wordSimPre)[0, 1]
    SpearCoef = stats.spearmanr(wordSimStd, wordSimPre).correlation
    SqrtCoef = np.sqrt(corrCoef * SpearCoef)
    return corrCoef, SpearCoef, SqrtCoef


file = open("./dataTest/COS960/COS960_all.txt", "r", encoding="utf-8")
cc = OpenCC("s2twp")
word1 = []
word2 = []
baseline = []
for line in file:
    w1, w2, valStr = line.strip().split()[0:3]
    word1.append(cc.convert(w1))
    word2.append(cc.convert(w2))
    baseline.append(valStr)
baseline = np.array(baseline)
pre_w = []
pre_f = []
pre_b = []
total_w = 0
total_f = 0
total_b = 0
corrCoef = [0, 0, 0]
SpearCoef = [0, 0, 0]
SqrtCoef = [0, 0, 0]

start = time.time()
for i in trange(len(word1)):
    pre_w.append(sim_w(word1[i], word2[i]))
end = time.time()
total_w = end - start
corrCoef[0], SpearCoef[0], SqrtCoef[0] = score(
    baseline.astype(float), np.array(pre_w).astype(float)
)
print("Word2Vec Result")
print("Time:" + f"{total_w}")
print("corrCoef:" + f"{corrCoef[0]}")
print("SpearCoef:" + f"{SpearCoef[0]}")
print("SqrtCoef:" + f"{SqrtCoef[0]}")

start = time.time()
for i in trange(len(word1)):
    pre_f.append(sim_f(word1[i], word2[i]))
end = time.time()
total_f = end - start
corrCoef[1], SpearCoef[1], SqrtCoef[1] = score(
    baseline.astype(float), np.array(pre_f).astype(float)
)

start = time.time()
for i in trange(len(word1)):
    pre_b.append(sim_b(word1[i], word2[i]))
end = time.time()
total_b = end - start
corrCoef[2], SpearCoef[2], SqrtCoef[2] = score(
    baseline.astype(float), np.array(pre_b).astype(float)
)

output_path = "tradeoff.txt"
with open(output_path, "w", encoding="utf-8") as output_file:
    output_file.write("Word2Vec Result\n")
    output_file.write("Time:" + f"{total_w}\n")
    output_file.write("corrCoef:" + f"{corrCoef[0]}\n")
    output_file.write("SpearCoef:" + f"{SpearCoef[0]}\n")
    output_file.write("SqrtCoef:" + f"{SqrtCoef[0]}\n\n")

    output_file.write("FastText Result\n")
    output_file.write("Time:" + f"{total_f}\n")
    output_file.write("corrCoef:" + f"{corrCoef[1]}\n")
    output_file.write("SpearCoef:" + f"{SpearCoef[1]}\n")
    output_file.write("SqrtCoef:" + f"{SqrtCoef[1]}\n\n")

    output_file.write("Bert Result\n")
    output_file.write("Time:" + f"{total_b}\n")
    output_file.write("corrCoef:" + f"{corrCoef[2]}\n")
    output_file.write("SpearCoef:" + f"{SpearCoef[2]}\n")
    output_file.write("SqrtCoef:" + f"{SqrtCoef[2]}\n")
