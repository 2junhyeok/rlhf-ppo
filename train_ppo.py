import torch
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from torch.utils.data import DataLoader

from safety_RM.reward_model import kcBERTRM, load_model

def build_dataset(tokenizer, dataset_name="MrBananaHuman/kor_ethical_question_answer"):
    ds = load_dataset(dataset_name)

    def preprocess(sample):
        prompt = sample["question"]
        encoded = tokenizer.encode_plus(
            prompt, 
            max_length=128, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        sample["input_ids"] = encoded["input_ids"].squeeze(0)
        sample["query"] = prompt
        return sample

    ds = ds.map(preprocess)
    ds.set_format(type="torch")
    return ds["train"]

def build_collator(tokenizer):
    def collator(data):
        queries = [d["query"] for d in data]
        batch_encoding = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        return {
            "input_ids": batch_encoding["input_ids"],
            "attention_mask": batch_encoding["attention_mask"],
            "query": queries
        }
    return collator

def build_reward_model():
    rm = kcBERTRM()
    rm = load_model(rm, "./safety_RM/checkpoints/final_model.pt", device="cuda")
    rm.eval()
    rm.to("cuda")
    return rm

def compute_rewards(rm, texts):
    with torch.no_grad():
        batch = rm.tokenize(texts, padding="max_length", truncation=True, max_length=128).to("cuda")
        rewards = rm(**batch).squeeze()
    return [r.detach().cpu() for r in rewards]

def main():
    model_name = "jeong-6/kogpt2-safety-finetuned"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>'
    )
    

    tokenizer.padding_side = "left"
    

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(tokenizer)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to("cuda")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to("cuda")
    collator = build_collator(tokenizer)
    dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collator)

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=2,
        ppo_epochs=1,
        seed=42,
        log_with=None,
        init_kl_coef=0.05,
        target_kl=0.1
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_ds,
        data_collator=collator
    )

    rm = build_reward_model()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    max_reward = float('-inf')
    total_iterations = 1000

    for iteration, batch in enumerate(dataloader):
        if iteration >= total_iterations:
            break

        query_tensors = [q.to("cuda") for q in batch["input_ids"]]
        queries = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]

        responses = []
        for q in query_tensors:
            output = model.generate(
                q.unsqueeze(0), 
                max_length=128, 
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
            response = output[0][q.size(0):]
            responses.append(response)

        response_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in responses]
        current_rewards = compute_rewards(rm, response_texts)
        avg_reward = sum([r.item() for r in current_rewards]) / len(current_rewards)

        stats = ppo_trainer.step(query_tensors, responses, current_rewards)

        if iteration % 100 == 0 or iteration == total_iterations - 1:
            print(f"Iteration {iteration}/{total_iterations} - Avg Reward: {avg_reward:.4f}")

        if avg_reward > max_reward:
            max_reward = avg_reward
            checkpoint_path = os.path.join(checkpoint_dir, f"model_iter_{iteration}_reward_{avg_reward:.4f}")
            model.save_pretrained(checkpoint_path)

    final_checkpoint_path = os.path.join(checkpoint_dir, "model_final")
    model.save_pretrained(final_checkpoint_path)

if __name__ == "__main__":
    main()