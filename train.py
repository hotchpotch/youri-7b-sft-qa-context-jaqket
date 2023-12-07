# %%
DEBUG = True

MODEL_NAME = "rinna/youri-7b-instruction"
OUTPUT_DIR = f"./output_v3_{MODEL_NAME.split('/')[-1]}"

if DEBUG:
    OUTPUT_DIR = OUTPUT_DIR + "_debug"

RESPONSE_MESSAGE = "応答"
RESPONSE_PROMPT = f"### {RESPONSE_MESSAGE}: \n"


def build_prompt(
    user_message: str,
    inputs: str | None = "",
    separator: str = "\n\n### ",
    response_message: str = RESPONSE_MESSAGE,
) -> str:
    system_message = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    prompt = system_message
    roles = ["指示", response_message]
    messages = [": \n" + user_message, ": \n"]

    if inputs:
        roles = ["指示", "入力", response_message]
        messages = [": \n" + user_message, ": \n" + inputs, ": \n"]

    for role, message in zip(roles, messages):
        prompt += separator + role + message
    return prompt


# %%
import pandas as pd
from IPython.display import display
import datasets

ds = datasets.load_dataset("hotchpotch/jaqket_v1_qa_wikija_context")
train_ds = ds["train"]  # type: ignore
valid_ds = ds["validation"]  # type: ignore

train_df = train_ds.data.to_pandas()  # type: ignore
valid_df = valid_ds.data.to_pandas()  # type: ignore

# context は list なので、 "\n" で結合する
train_df["context"] = train_df["context"].apply(lambda x: "\n".join(x) + "\n")
valid_df["context"] = valid_df["context"].apply(lambda x: "\n".join(x) + "\n")

train_ds = datasets.Dataset.from_pandas(train_df)
valid_ds = datasets.Dataset.from_pandas(valid_df)

display(train_ds.shape, valid_ds.shape)

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    use_flash_attention_2=True,
)


# %%
def formatting_prompts_func(example, return_dict=False):
    output_texts = []
    for i in range(len(example["question"])):
        question = example["question"][i]
        context = example["context"][i]
        answer = example["answer"][i]
        text = build_prompt(question, context) + answer
        output_texts.append(text)
    if return_dict:
        return {"input_text": output_texts}
    else:
        return output_texts


# %%
# # token長がどれぐらいかを見る
# import matplotlib.pyplot as plt


# def token_count(text):
#     return len(tokenizer.tokenize(text, add_special_tokens=False))


# train_ds = datasets.Dataset.from_pandas(valid_df)
# train_ds = train_ds.map(
#     formatting_prompts_func, batched=True, fn_kwargs={"return_dict": True}
# )
# input_text_token_count = map(token_count, train_ds["input_text"])
# input_text_token_count = list(input_text_token_count)

# plt.hist(input_text_token_count, bins=100)
# plt.show()

# %%
# response_templateのtoken_idがちゃんと含まれているかの確認


def print_tokens_with_ids(txt):
    tokens = tokenizer.tokenize(txt, add_special_tokens=False)
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    print(list(zip(tokens, token_ids)))


prompt = build_prompt("質問です", "文脈です") + "回答です"

print(prompt)
print_tokens_with_ids(prompt)
print_tokens_with_ids(RESPONSE_PROMPT)

# %%
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
response_template_ids = tokenizer.encode(RESPONSE_PROMPT, add_special_tokens=False)
while len(response_template_ids) > 0:
    # prompt_ids に部分一致するか
    matched = False
    for i in range(len(prompt_ids)):
        if prompt_ids[i : i + len(response_template_ids)] == response_template_ids:
            matched = True
            break
    if matched:
        break
    response_template_ids = response_template_ids[1:]
    if len(response_template_ids) == 0:
        raise ValueError("response_template_ids is not included in prompt_ids")
# 実際にマッチした部分の token_ids を使う
print(response_template_ids)

# %%
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

if DEBUG:
    train_ds = train_ds.shuffle(seed=42).select(range(50))
    valid_ds = valid_ds.shuffle(seed=42).select(range(10))
else:
    # 全件 valid すると時間がかかるので、目安程度の一部だけにする
    valid_ds = valid_ds.shuffle(seed=42).select(range(50))

# %%
tokenizer.pad_token = tokenizer.eos_token

# %%
from peft import LoraConfig  # type: ignore
import os
from transformers import TrainerCallback, TrainingArguments


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

callbacks = [PeftSavingCallback()]
training_args = TrainingArguments(
    learning_rate=5e-4,
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    optim="paged_adamw_32bit",
    evaluation_strategy="steps",
    lr_scheduler_type="cosine",
    logging_dir="./logs",
    logging_steps=2,
    save_steps=5,
    eval_steps=30,
    warmup_steps=10,
    max_grad_norm=0.3,
    weight_decay=0.01,
    save_total_limit=1,
    neftune_noise_alpha=5,
)

if DEBUG:
    training_args.learning_rate = 2e-4
    training_args.gradient_accumulation_steps = 4
    training_args.logging_steps = 2
    training_args.eval_steps = 13
    training_args.warmup_steps = 2


trainer = SFTTrainer(
    model,
    train_dataset=train_ds,  # type: ignore
    eval_dataset=valid_ds,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=2048,
    peft_config=peft_config,
    callbacks=callbacks,  # type: ignore
    args=training_args,
)

trainer.train()  # type: ignore

# %%
from shutil import rmtree
from pathlib import Path

trainer.save_model(output_dir=OUTPUT_DIR)

for path in Path(training_args.output_dir).glob("checkpoint-*"):
    if path.is_dir():
        rmtree(path)

# %%
del trainer
del model
torch.cuda.empty_cache()

# %%
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    use_flash_attention_2=True,
)

# %%
from peft import PeftModel  # type: ignore

model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

valid_df = valid_ds.data.to_pandas()

# %%
target_n = 4
prompt_template = build_prompt(
    valid_df["question"][target_n], valid_df["context"][target_n]
)
print(prompt_template)
print("正解 -> ", valid_df["answer"][target_n])

# %%
token_ids = tokenizer.encode(
    prompt_template, add_special_tokens=False, return_tensors="pt"
)

with torch.no_grad():
    output_ids = model.generate(
        input_ids=token_ids.to(model.device),  # type: ignore
        max_new_tokens=24,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
output = tokenizer.decode(output_ids.tolist()[0])  # type: ignore

print(output)


# %%
def qa(model, tokenizer, question, context, build_prompt_fn=build_prompt):
    prompt = build_prompt_fn(question, context)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=token_ids.to(model.device),  # type: ignore
            max_new_tokens=24,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0])
    # prompt を取り除く
    output = output.replace(prompt, "")
    # eos_token 以降を取り除く
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()


print(
    qa(
        model,
        tokenizer,
        valid_df["question"][target_n],
        valid_df["context"][target_n],
        build_prompt_fn=build_prompt,
    )
)

# %%
# valid_df での正解率を測る
from tqdm import tqdm

valid_df = valid_df.reset_index(drop=True)
for i in tqdm(range(len(valid_df))):
    question = valid_df["question"][i]
    context = valid_df["context"][i]
    answer = valid_df["answer"][i]
    pred = qa(model, tokenizer, question, context, build_prompt_fn=build_prompt)
    valid_df.loc[i, "pred"] = pred

# %%
# 完全一致の正解率を表示
import wandb

valid_df["is_correct"] = valid_df["answer"] == valid_df["pred"]
# print("完全一致の正解率")
# display(valid_df["is_correct"].mean())
wandb.log({"完全一致の正解率": valid_df["is_correct"].mean()})

# %%
wandb.Table(
    dataframe=valid_df[["question", "answer", "pred", "is_correct", "context"]][
        valid_df["is_correct"] == False
    ]
)

# %%
# 部分一致の正解率を表示
valid_df["is_correct"] = valid_df.apply(lambda x: x["answer"] in x["pred"], axis=1)
# print("部分一致の正解率:")
# print(valid_df["is_correct"].mean())
wandb.log({"部分一致の正解率": valid_df["is_correct"].mean()})

# %%
valid_df[["question", "answer", "pred", "is_correct"]].head(100)
# 間違ったものだけを表示
wandb.Table(
    dataframe=valid_df[["question", "answer", "pred", "is_correct", "context"]][
        valid_df["is_correct"] == False
    ]
)

# %%
valid_df.shape

# %%
# merge_model = model.merge_and_unload()

# %%
# merge_model.save_pretrained("./output_merge_model/")
