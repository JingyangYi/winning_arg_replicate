import subprocess
import json
import pandas as pd
from tqdm import tqdm


def create_prompt(data):
    input = data["input"]
    instruction = data["instruction"]
    prompt = f"Instruction: {instruction}\n  Input: {input} \n Response:"
    return prompt


def get_ollama_response(prompt, model):
    try:
        result = subprocess.run(["ollama", "run", model],
                                input=prompt.encode('utf-8'),
                                capture_output=True,
                                check=True
                                )
        response = result.stdout.decode('utf-8').strip()
        return response
    except subprocess.CalledProcessError as e:
        print(f"Error {e}")
        return None


def batch_inference(model, dataset_file, batch_size):
    data = []
    with open(dataset_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    n_batches = (len(df) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Batch Inferring"):
        start_point = i * batch_size
        end_point = (i + 1) * batch_size
        batch = df[start_point: end_point]

        with open('llama_results.jsonl', "w") as f:
            for idx, row in batch.iterrows():
                prompt = create_prompt(row)
                response = get_ollama_response(prompt, model)
                body = {
                    "actual": row["output"],
                    "ft_llama_prediction": response
                }
                json.dump(body, f)
                f.write("\n")

        break


modelfile_content = """
FROM /Users/yijingyang/Desktop/unsloth.Q4_K_M.gguf

PARAMETER temperature 1
PARAMETER num_ctx 4000

SYSTEM You're a semantic analyst. Judging from the opinion statement, do you think he/she is resistant or malleable to persuasion?
"""

model_name = "ft_llama_with_CMV"

with open("modelfile.txt", "w") as f:
    f.write(modelfile_content)

try:
    result = subprocess.run(["ollama", 'create', model_name, "--file", "modelfile.txt"],
                            capture_output=True,
                            check=True)
    print(result.stdout.decode("utf-8"))

except subprocess.CalledProcessError as e:
    print(f"Error loading model: {e}")

batch_inference(model_name, "op_test.jsonl", batch_size=1)