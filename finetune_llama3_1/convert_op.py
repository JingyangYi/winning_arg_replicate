import tarfile
import json
from bz2 import BZ2File


def extract_jsonlist(tar_path, file_name):
    with tarfile.open(tar_path, mode='r') as tar:
        bz2_file = tar.extractfile(file_name)
        return [json.loads(line.decode('utf-8')) for line in BZ2File(bz2_file)]


# File paths
fname = "../cmv.tar.bz2"
train_fname = "op_task/train_op_data.jsonlist.bz2"
test_fname = "op_task/heldout_op_data.jsonlist.bz2"

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


original_posts_train = create_balanced_dataset(original_posts_train, n_samples=500)
original_posts_test = create_balanced_dataset(original_posts_test, n_samples=100)


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

create_jsonl(original_posts_train, 'op_train.jsonl', instruction)
create_jsonl(original_posts_test, 'op_test.jsonl', instruction)
