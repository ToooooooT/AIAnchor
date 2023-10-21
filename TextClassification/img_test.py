import torch
import argparse
from transformers import RobertaTokenizer
from tqdm import tqdm
from multiprocessing import Process, Queue
import json
import requests
import os
from similar import similarity

parser = argparse.ArgumentParser("test")
parser.add_argument("--text", type=str, default=None, help="text")
parser.add_argument("--text_path", type=str, default=None, help="text path")
parser.add_argument(
    "--ckpt", type=str, default="model_1e-05_64.pth", help="checkpoint path"
)
parser.add_argument(
    "--save_path", type=str, default="./best_img/img.jpg", help="checkpoint path"
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

file_path = args.text_path
text = args.text
# text = "《小天使與小惡魔的誠信對話》　南投廉政志工團展開校園巡演"
if file_path != None:
    try:
        with open(file_path, "r") as file:
            text = file.read()
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")

checkpoint_path = args.ckpt
model = torch.load(checkpoint_path).to(device)

model.eval()
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
encoding = tokenizer(
    text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
)
with torch.no_grad():
    outputs = model(
        encoding["input_ids"].to(device),
        attention_mask=encoding["attention_mask"].to(device),
        labels=torch.tensor(0, device=device),
    )
    logits = outputs.logits

best_cat = torch.argmax(logits, dim=1).cpu().item() + 1
print(f"Category: {best_cat}")


def load_data(path):
    with open(path, "r") as file_category:
        with tqdm(
            total=1, unit="JSON files", desc="Loading Category Data", colour="CYAN"
        ) as pbar:
            data = json.load(file_category)
            pbar.update(1)
    return data


# path = "./dataset/data_img.json"
path = "/home/toooot/ETtoday/TextClassification/dataset/data_cat1_img.json"
data = load_data(path)

# title_list = list(data[str(best_cat)]['7'].items())
title_list = list(data[str(best_cat)].items())

best = 0
best_url = None
num_processes = 8

if __name__ == "__main__":
    processes = []
    chunk_size = len(title_list) // num_processes
    q = Queue()

    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_processes - 1 else len(title_list)
        p = Process(target=similarity, args=(q, start, end, title_list, text))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = []
    while not q.empty():
        results.append(q.get())

    # Find the best similarity result from the list of results
    index = 0
    max_sim = 0
    for i, (id, simi) in enumerate(results):
        if simi > max_sim:
            max_sim = simi
            index = id

    best_url = title_list[index][1]

    def web_crab(save_path, image_urls):
        response = requests.get(f"https:{image_urls}")
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
        else:
            print("Download Fail!")

    def get_img():
        if not os.path.exists("./best_img/"):
            os.makedirs("./best_img/")
        web_crab(args.save_path, best_url)

    get_img()
