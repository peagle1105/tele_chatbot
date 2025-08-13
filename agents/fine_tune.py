import unsloth

from unsloth import FastLanguageModel, to_sharegpt
from datasets import load_dataset, Dataset
import torch
from trl import SFTTrainer
from transformers.training_args import TrainingArguments
from unsloth import is_bfloat16_supported
import pandas as pd

# Define parameters
model_name = "/mnt/d/HOPT/Agent_build/tele_agent/models/fine_tuned_model"
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Declare model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    dtype=None,
    device_map="auto",
)

# Load dataset
dataset_vipubmed = load_dataset("VietAI/vi_pubmed", split = "pubmed22")

def format_vipubmed(example):
    instruction = "Hãy đọc văn bản tiếng anh do người dùng cung cấp và dịch sang tiếng việt."
    en_text = example['en']
    input = f"Bạn hãy giúp tôi dịch văn bản sau đây sang tiếng Việt: {en_text}"
    output = f"Văn bản sau khi được dịch là:\n\t{example['vi']}"
    return {
        'instruction': instruction,
        'input': input,
        'output': output,
    }

dataset_vipubmed = dataset_vipubmed.map(format_vipubmed)

dataset = pd.DataFrame(dataset_vipubmed[:100000])

dataset = to_sharegpt(
    Dataset.from_pandas(dataset),
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,
)

# Create fine-tune model
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules= ["q_proj", "k_proj", "v_proj", 
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha= 16,
    lora_dropout= 0,
    bias = "none",
    use_gradient_checkpointing= "unsloth",
    random_state= 3407,
    use_rslora= False,
    loftq_config= None
)

def formatting_func(batch):
    # batch is a dict: {"conversations": [ [...], [...], ... ]}
    formatted_texts = []

    system_prompt = "You are a helpful medical assistant. Answer based on the patient's description."

    for conversation in batch["conversations"]:  # Each `conversation` is a list of {'from': ..., 'value': ...}
        text = "<|begin_of_text|>"

        for msg in conversation:
            if not isinstance(msg, dict):
                continue
            if "from" not in msg or "value" not in msg:
                continue

            role = msg["from"]
            content = msg["value"].strip()

            if role == "human":
                if "If you are a doctor" in content and "Your input is:" in content:
                    try:
                        user_input = content.split("Your input is:\n", 1)[1]
                    except IndexError:
                        user_input = content
                    text += f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
                else:
                    text += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
            elif role == "gpt":
                text += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"

        formatted_texts.append(text)

    return formatted_texts  # List of strings, one per example

trainer = SFTTrainer(
    model = model,
    processing_class= tokenizer,
    train_dataset= dataset,
    peft_config= model.peft_config,
    formatting_func = formatting_func,
    args= TrainingArguments(
        output_dir= "./output",
        per_device_train_batch_size= 2,
        gradient_accumulation_steps= 4,
        warmup_steps= 5,
        max_steps= 60,
        num_train_epochs= 1,
        learning_rate= 2e-4,
        fp16= not is_bfloat16_supported(),
        bf16= is_bfloat16_supported(),
        logging_steps= 1,
        optim="adamw_8bit",
        weight_decay= 0.01,
        lr_scheduler_type= "linear",
        seed= 3407,
    )
)
trainer.train()

# Save model
model.save_pretrained("../models/fine_tuned_model")
tokenizer.save_pretrained("../models/fine_tuned_model")