import pandas as pd
import re
from tqdm import tqdm
import json



def hashid(cat_lv1_id, cat_lv2_id):
    return cat_lv1_id * 1000 + cat_lv2_id


def export_json(dict1, name):
    # Convert dictionary to JSON string
    json_data = json.dumps(dict1)
    # Save JSON string to a file
    with open(f'./dataset/{name}', 'w') as json_file:
        json_file.write(json_data)
    pass

if __name__ == '__main__':
    data = pd.read_csv('./dataset/ettoday_news.tsv', delimiter='\t')

    pattern = r'<img\+src=\\"([^"]+)\\"\+alt=\\"([^"]+)\\"\+width=\\"(\d+)\\"\+height=\\"(\d+)\\"\+title=\\"([^"]+)\\"\+/>'
    my_dict = {}
    df_cleaned = data.dropna(subset=['cat_lv1_id', 'cat_lv2_id', 'content'])
    text = []
    label = []

    tmp = data[['cat_lv1_id', 'cat_lv2_id']].dropna()
    unique_tuples = list(tmp.drop_duplicates().to_records(index=False))
    threshold = 5000
    tmp = data.groupby(['cat_lv1_id']).count()['title']
    cat_lv1_id_split = list(tmp[tmp > threshold].index)


    for i in tqdm(range(df_cleaned.shape[0])):
        row = df_cleaned.iloc[i]
        title = row['title']
        cat_lv1_id = int(row['cat_lv1_id'])
        cat_lv2_id = int(row['cat_lv2_id'])
        content = row['content']

        # Use re.search to find the pattern in the string
        matches = re.finditer(pattern, content)

        for m in matches:
            # Extract information from the match
            src = m.group(1)
            alt = m.group(2)
            width = m.group(3)
            height = m.group(4)
            title = m.group(5)
            title = re.sub(r'（\u5716.*）', '', title)
            title = re.sub(r'▲', '', title)
            title = re.sub(r'▼', '', title)
            title = re.sub(r'\+', '', title)

            if cat_lv1_id not in my_dict.keys():
                my_dict[cat_lv1_id] = {}

            if cat_lv2_id not in my_dict[cat_lv1_id].keys() and cat_lv1_id in cat_lv1_id_split: 
                my_dict[cat_lv1_id][cat_lv2_id] = {}

            if cat_lv1_id in cat_lv1_id_split: 
                my_dict[cat_lv1_id][cat_lv2_id][title] = src
            else:
                my_dict[cat_lv1_id][title] = src
            text.append(title)
            label.append(hashid(cat_lv1_id, cat_lv2_id))

    export_json(my_dict, 'data_img.json')
    train_data = pd.DataFrame({'text': text, 'label': label})
    train_data.to_csv('data_subcat.csv', index=False)

    id_index_map = {}
    index_id_map = {}
    i = 0
    for tup in unique_tuples:
        if tup[0] in cat_lv1_id_split:
            index_id_map[int(i)] = tup
            id_index_map[int(hashid(tup[0], tup[1]))] = i
            i += 1
        elif hashid(tup[0], tup[1]) not in id_index_map.keys():
            index_id_map[int(i)] = (tup[0],)
            id_index_map[int(hashid(tup[0], 0))] = i
            i += 1

    export_json(id_index_map, 'id_to_index.json')
    export_json(index_id_map, 'index_to_id.json')