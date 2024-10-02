import tarfile
import json
import os.path
from urllib import request
from io import BytesIO
from bz2 import BZ2File

# File paths
fname = "cmv.tar.bz2"
train_fname = "pair_task/train_pair_data.jsonlist.bz2"
test_fname = "pair_task/heldout_pair_data.jsonlist.bz2"
url = "https://chenhaot.com/data/cmv/" + fname

# download if not exists
if not os.path.isfile(fname):
    f = BytesIO()
    with request.urlopen(url) as resp, open(fname, 'wb') as f_disk:
        data = resp.read()
        f_disk.write(data)  # save to disk too
        f.write(data)
        f.seek(0)


def extract_jsonlist(tar_path, file_name):
    with tarfile.open(tar_path, mode='r') as tar:
        bz2_file = tar.extractfile(file_name)
        return [json.loads(line.decode('utf-8')) for line in BZ2File(bz2_file)]


print("Extracting jsonl files.")
# Extract and deserialize the JSON lists
pairs_train = extract_jsonlist(fname, train_fname)
pairs_test = extract_jsonlist(fname, test_fname)


def create_nonoverlapping_pairs(data, n_sample):
    op_names = []
    dataset = []
    unique_count = 0
    for line in data:
        if line["op_name"] not in op_names:
            op_names.append(line["op_name"])
            dataset.append(line)
            unique_count += 1
        if unique_count >= n_sample:
            break
    return dataset


n_samples_train = 1000
n_samples_test = 200
print(
    f"Creating non-overlapping dataset. Number of training set: {n_samples_train}, number of test set: {n_samples_test}")
pairs_train = create_nonoverlapping_pairs(pairs_train, n_samples_train)
pairs_test = create_nonoverlapping_pairs(pairs_test, n_samples_test)


def cleanup(cmv_post):
    lines = [line for line in cmv_post.splitlines()
             if not line.lstrip().startswith("&gt;")
             and not line.lstrip().startswith("____")
             and "edit" not in " ".join(line.lower().split()[:2])
             ]
    return "\n".join(lines)


def creat_op_reply_json(data, file_path1, file_path2, instruction):
    import random
    random.seed(42)
    with open(file_path1, "w") as f1, open(file_path2, "w") as f2:
        for idx, line in enumerate(data):
            op_text = cleanup(line["op_text"])
            op_title = line["op_title"]
            reply_pos = line['positive']['comments'][0]['body']
            reply_neg = line['negative']['comments'][0]['body']
            replies = [("positive", reply_pos), ("negative", reply_neg)]
            random.shuffle(replies)

            input = f"Original Post:\n{op_title}\n{op_text}\nFirst Reply:\n{replies[0][1]}\nSecond Reply:\n{replies[1][1]}"
            output = "first" if replies[0][0] == "positive" else "second"

            alpaca_content = {
                "input": input,
                "output": output,
                "instruction": instruction
            }

            gpt_content = {
                "custom_id": f"request={idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-3.5-turbo-0125",
                         "messages": [{"role": "system", "content": instruction},
                                      {"role": "user", "content": input}
                                      ],
                         "max_tokens": 1000},
            }
            json.dump(alpaca_content, f1)
            json.dump(gpt_content, f2)
            f1.write("\n")
            f2.write("\n")


instruction = "This is a conversation from an online discussion community. The first was a poster who posted an opinion, and the next two replies were each trying to convince the poster to revise his opinion. The two responses were similar, but one managed to convince the poster and the other didn't. Now judge which response succeeded in persuading. Answer first or second only."

train_file_path1 = "finetune_llama3/finetune_datasets/pairs_train_alpaca.jsonl"
train_file_path2 = 'prompts_datasets/pairs_train_gpt.jsonl'
test_file_path1 = "finetune_llama3/finetune_datasets/pairs_test_alpaca.jsonl"
test_file_path2 = 'prompts_datasets/pairs_test_gpt.jsonl'

creat_op_reply_json(pairs_train, train_file_path1, train_file_path2, instruction)
creat_op_reply_json(pairs_test, test_file_path1, test_file_path2, instruction)
print("Created datasets for llama finetuning.")
print("Created datasets for GPT3.5 prompts.")