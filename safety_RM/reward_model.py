import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class kcBERTRM(nn.Module):
    def __init__(self, model_name="beomi/kcbert-base"):
        super(kcBERTRM, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        pooled_output = outputs.pooler_output
        reward = self.reward_head(pooled_output)
        return reward
    
    def tokenize(self, texts, max_length=128, padding=True, truncation=True):
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device=None):
    model.load_state_dict(torch.load(path, map_location=device))
    return model