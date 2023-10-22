from gensim.models import Word2Vec
from multiprocessing import Queue
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

model_name_ = "bert-base-chinese"
tokenizer_ = AutoTokenizer.from_pretrained(model_name_)
model_b_ = AutoModel.from_pretrained(model_name_)

# Hugging Face Transformer Bert
def sim(text1, text2):

    inputs = tokenizer_(
        [text1, text2], return_tensors="pt", padding=True, truncation=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model_b_(**inputs)

    embeddings1 = outputs.last_hidden_state[0].mean(dim=0)
    embeddings2 = outputs.last_hidden_state[1].mean(dim=0)

    similarity = (embeddings1.dot(embeddings2)) / (
        torch.norm(embeddings1) * torch.norm(embeddings2)
    )
    return similarity.item()

# def sim(text1, text2):
#     seg_text1 = list(jieba.cut(text1))
#     seg_text2 = list(jieba.cut(text2))
#     model = Word2Vec(
#         [seg_text1, seg_text2], vector_size=100, window=5, min_count=1, sg=0
#     )
#     model.train([seg_text1, seg_text2], total_examples=2, epochs=10)
#     vec1 = np.mean([model.wv[word] for word in seg_text1 if word in model.wv], axis=0)
#     vec2 = np.mean([model.wv[word] for word in seg_text2 if word in model.wv], axis=0)
#     similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#     return similarity


def similarity(q: Queue, start, end, title_list, text):
    max_prob = 0
    ret = 0
    for i in tqdm(range(start, end)):
        try:
            prob = sim(title_list[i][0], text)
            if prob > max_prob:
                ret = i
                max_prob = prob
        except:
            pass
    q.put((ret, max_prob))
