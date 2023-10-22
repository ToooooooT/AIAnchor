from gensim.models import Word2Vec
from multiprocessing import Queue
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast
import torch
import jieba

model_name_ = "bert-base-chinese"
# tokenizer_ = AutoTokenizer.from_pretrained(model_name_)
tokenizer_ = BertTokenizerFast.from_pretrained(model_name_)
model_b_ = AutoModel.from_pretrained(model_name_)

# Hugging Face Transformer Bert
# def sim(text1, text2):
#     words = list(jieba.cut(text2))
#     text2 = max(words, key=len)

#     inputs = tokenizer_(
#         [text1, text2], return_tensors="pt", padding=True, truncation=True
#     ).to("cuda")

#     with torch.no_grad():
#         outputs = model_b_(**inputs)

#     embeddings1 = outputs.last_hidden_state[0].mean(dim=0)
#     embeddings2 = outputs.last_hidden_state[1].mean(dim=0)

#     similarity = (embeddings1.dot(embeddings2)) / (
#         torch.norm(embeddings1) * torch.norm(embeddings2)
#     )
#     return similarity.item()

# nigo
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

# rara
def sim(text1, text2):
    textList1 = text1.split("，")
    textList2 = text2.split("，")

    vec1 = []
    vec2 = []

    inputs = tokenizer_(
        [textList1], return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    with torch.no_grad():
        outputs = model_b_(**inputs)
    vec1 = [outputs.last_hidden_state[i].mean(dim=0) for i in range(textList1)]
    average_tensor1 = torch.mean(torch.stack(vec1), dim=0)


    inputs = tokenizer_(
        [textList2], return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    with torch.no_grad():
        outputs = model_b_(**inputs)
    vec2 = [outputs.last_hidden_state[i].mean(dim=0) for i in range(textList2)]
    average_tensor2 = torch.mean(torch.stack(vec2), dim=0)

    similarity = (average_tensor1.dot(average_tensor2)) / (
        torch.norm(average_tensor1) * torch.norm(average_tensor2)
    )
    
    return similarity.item()


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
