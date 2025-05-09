import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
from reward_model import kcBERTRM, save_model
from dataset import EthicalQADataset, get_splits
from tqdm import tqdm

def train_model(
    train_dataset,
    val_dataset,
    model,
    device,
    batch_size=16,
    epochs=3,
    lr=1e-5,
    save_dir="checkpoints",
    save_steps=1000
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch:{epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            rewards = batch["reward"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = torch.nn.functional.binary_cross_entropy(outputs.squeeze(-1), rewards)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % save_steps == 0:
                save_path = os.path.join(save_dir, f"checkpoint_{step}step.pt")
                save_model(model, save_path)
        
        print(f"Epoch:{epoch+1}/{epochs}, AVG_train_loss:{total_loss/len(train_loader):.4f}")
        
        # val
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                rewards = batch["reward"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), rewards)
                total_val_loss += loss.item()
        
        print(f"Epoch:{epoch+1}/{epochs}, AVG_val_loss:{total_val_loss/len(val_loader):.4f}")
    
    final_save_path = os.path.join(save_dir, "final_model.pt")
    save_model(model, final_save_path)

def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    train_split, val_split = get_splits(seed=42, val_ratio=0.2)
    train_dataset = EthicalQADataset(train_split)
    val_dataset = EthicalQADataset(val_split)
    
    model = kcBERTRM()
    
    
    train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        device=device,
        batch_size=16,
        epochs=3,
        lr=2e-5,
        save_dir="checkpoints",
        save_steps=1000
    )

if __name__=="__main__":
    main()