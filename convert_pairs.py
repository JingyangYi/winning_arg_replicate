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


def creat_op_reply_json(data, file_path, instruction):
    import random
    random.seed(42)
    with open(file_path, "w") as f:
        for line in data:
            op_text = cleanup(line["op_text"])
            op_title = line["op_title"]
            reply_pos = line['positive']['comments'][0]['body']
            reply_neg = line['negative']['comments'][0]['body']
            replies = [("positive", reply_pos), ("negative", reply_neg)]
            random.shuffle(replies)

            input = f"Original Post:\n{op_title}\n{op_text}\nReply 1:\n{replies[0][1]}\nReply 2:\n{replies[1][1]}"
            output = "reply 1" if replies[0][0] == "positive" else "reply 2"
            request_content = {
                "input": input,
                "output": output,
                "instruction": instruction
            }
            json.dump(request_content, f)
            f.write("\n")


instruction = "This is a conversation from an online discussion community. The first was a poster who posted an opinion, and the next two replies were each trying to convince the poster to revise his opinion. The two responses were similar, but one managed to convince the poster and the other didn't. Now judge which response succeeded in persuading. Answer reply 1 or reply 1 only。"

creat_op_reply_json(pairs_train, "finetune_llama3_1/pairs_train_alpaca.jsonl", instruction)
creat_op_reply_json(pairs_test, "finetune_llama3_1/pairs_test_alpaca.jsonl", instruction)
print("Created datasets for llama finetuning.")


def create_gpt_prompts(data, file_path, instruction):
    with open(file_path, 'w') as f:
        for idx, line in enumerate(data):
            op_text = cleanup(line["op_text"])
            op_title = line["op_title"]
            reply_pos = line['positive']['comments'][0]['body']
            reply_neg = line['negative']['comments'][0]['body']
            replies = [("positive", reply_pos), ("negative", reply_neg)]
            prompt = f"Original Post:\n{op_title}\n{op_text}\nReply 1:\n{replies[0][1]}\nReply 2:\n{replies[1][1]}"
            request_content = {
                "custom_id": f"request={idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-3.5-turbo-0125",
                         "messages": [{"role": "system", "content": instruction},
                                      {"role": "user", "content": prompt}
                                      ],
                         "max_tokens": 1000},
            }
            json.dump(request_content, f)
            f.write('\n')


create_gpt_prompts(pairs_train, 'pairs_train_gpt.jsonl', instruction)
create_gpt_prompts(pairs_test, 'pairs_test_gpt.jsonl', instruction)
print("Created datasets for GPT3.5 prompts.")