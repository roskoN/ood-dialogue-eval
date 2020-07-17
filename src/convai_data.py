import json

import requests
from tqdm.autonotebook import tqdm

convai1_data = requests.get("http://convai.io/2017/data/train_full.json").json()
print(len(convai1_data))
convai2_data = requests.get(
    "http://convai.io/data/summer_wild_evaluation_dialogs.json"
).json()
print(len(convai2_data))

for dial in tqdm(convai1_data):
    quality = sum(
        [participant_score["quality"] for participant_score in dial["evaluation"]]
    ) / len(dial["evaluation"])
    dial["quality"] = quality
    utterances = [thread_line["text"] for thread_line in dial["thread"]]
    dial["utterances"] = utterances
    dial["predictions"] = dict()
    dial["id"] = str(dial["dialogId"])
    dial["dataset"] = 1

convai1_data = [dial for dial in convai1_data if len(dial["utterances"]) > 2]
print(len(convai1_data))

for dial in tqdm(convai2_data):
    dial["quality"] = dial["eval_score"]
    utterances = [thread_line["text"] for thread_line in dial["dialog"]]
    dial["utterances"] = utterances
    dial["predictions"] = dict()
    dial["id"] = str(dial["dialog_id"])
    dial["dataset"] = 2

convai2_data = [dial for dial in convai2_data if len(dial["utterances"]) > 2]
print(len(convai2_data))

for data_idx, data in enumerate([convai1_data, convai2_data], 1):
    data_fout = open(f"./data/convai/convai{data_idx}_data.tsv", "wt")
    idx_fout = open(f"./data/convai/convai{data_idx}_idx.tsv", "wt")
    data_list = list()
    for dialogue in data:
        for utt1, utt2 in zip(dialogue["utterances"], dialogue["utterances"][1:]):
            utt1 = utt1.strip().replace('\n', ' ')
            utt2 = utt2.strip().replace('\n', ' ')
            d_id = dialogue["id"]
            data_fout.write(f"1.0 {utt1}\t1.0 {utt2}\n")
            idx_fout.write(d_id + "\n")
    data_fout.close()
    idx_fout.close()

    with open(f"./data/convai/convai{data_idx}.json", "wt") as fout:
        json.dump(obj=data, fp=fout, indent=4)
