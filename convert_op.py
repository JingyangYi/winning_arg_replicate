import tarfile
import json
import os.path
from urllib import request
from io import BytesIO
from bz2 import BZ2File

# File paths
fname = "cmv.tar.bz2"
train_fname = "op_task/train_op_data.jsonlist.bz2"
test_fname = "op_task/heldout_op_data.jsonlist.bz2"
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
original_posts_train = extract_jsonlist(fname, train_fname)
original_posts_test = extract_jsonlist(fname, test_fname)


def cleanup(cmv_post):
    lines = [line for line in cmv_post.splitlines()
             if not line.lstrip().startswith("&gt;")
             and not line.lstrip().startswith("____")
             and "edit" not in " ".join(line.lower().split()[:2])
             ]
    return "\n".join(lines)


def clean_dataset(dataset):
    for i in range(len(dataset)):
        dataset[i]['selftext'] = cleanup(dataset[i]['selftext'])
    return dataset


print("Cleaning original dataset texts.")
original_posts_train = clean_dataset(original_posts_train)
original_posts_test = clean_dataset(original_posts_test)


def create_balanced_dataset(dataset, n_samples):
    import pandas as pd
    df = pd.DataFrame(dataset, columns=['title', 'delta_label', 'name', 'selftext'])
    is_malleable = df['delta_label']
    n_false, n_true = is_malleable.value_counts()
    n_samples = n_samples
    malleable = df[is_malleable].sample(n=n_samples,
                                        random_state=42)
    not_malleable = df[~is_malleable].sample(n=n_samples,
                                             random_state=42)
    df_balanced = pd.concat([malleable, not_malleable])
    # shuffle the dataframe
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    return df_balanced


train_n_samples = 1000
test_n_samples = 200
print(f"Creating balanced dataset with {train_n_samples} train samples and {test_n_samples} test samples.")
original_posts_train = create_balanced_dataset(original_posts_train, n_samples=train_n_samples//2)
original_posts_test = create_balanced_dataset(original_posts_test, n_samples=test_n_samples//2)


def create_jsonl(data, filename, instruction):
    with open(filename, 'w') as f:
        for _, line in data.iterrows():
            output = "malleable" if line["delta_label"] else "resistant"
            request_content = {
                "input": f"{line['title']} + \n + {line['selftext']}",
                "output": output,
                "instruction": instruction
            }
            json.dump(request_content, f)
            f.write('\n')


instruction = ("You're a semantic analyst. Judging from the opinion statement, do you think he/she is resistant or "
               "malleable to persuasion?")


print("Creating jsonl files for Llama finetuning.")
create_jsonl(original_posts_train, 'finetune_llama3_1/finetune_datasets/op_train_alpaca.jsonl', instruction)
create_jsonl(original_posts_test, 'finetune_llama3_1/finetune_datasets/op_test_alpaca.jsonl', instruction)


def create_gpt_prompts(data, filename, prefix):
    with open(filename, 'w') as f:
        for idx, row in data.iterrows():
            prompt = f"{row['title']} + \n + {row['selftext']}"
            request_content = {
            "custom_id" : f"request={idx}",
            "method" : "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-3.5-turbo-0125",
                     "messages": [{"role": "system", "content": prefix},
                                  {"role": "user", "content": prompt}
                                  ],
                     "max_tokens": 1000},
            }
            json.dump(request_content, f)
            f.write('\n')


prefix = "You're a semantic analyst. Now I will show you a person's opinion statement. We know that the person publicly announced his/her argument and encouraged other people to challenge it. Judging from the following context, do you think he/she is resistant or malleable to persuasion? Answer only with 'malleable' or 'resistant'. \n text: \n"

print("Creating jsonl files for GPT3.5 prompts.")
create_gpt_prompts(original_posts_train, 'prompts_datasets/op_train_gpt.jsonl', prefix)
create_gpt_prompts(original_posts_test, 'prompts_datasets/op_test_gpt.jsonl', prefix)