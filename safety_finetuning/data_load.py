from datasets import load_dataset
import pandas as pd


"""
https://huggingface.co/datasets/MrBananaHuman/kor_ethical_question_answer 에서 데이터셋 불러온 후 전처리 후 fine tuning해서 G_SFT를 만드는 게 목표입니다

데이터셋 형태는 {category:"성적내용"
                question:"왜 남자들은 여자를 쳐다보나요?"
                answer:"~~~"
                label: 0 or 1} 0이 정답 
"""

import pandas as pd
from datasets import load_dataset

ds = load_dataset("MrBananaHuman/kor_ethical_question_answer")
df = pd.DataFrame(ds['train'])

filtered_df = df[df['label'] == 0][['question', 'answer']]
filtered_df = filtered_df[
    (filtered_df['question'].str.len() <= 250) & (filtered_df['answer'].str.len() <= 250)
]
filtered_df_500 = filtered_df.sample(n=500, random_state=42)
filtered_df_500 = filtered_df_500.rename(columns={'question': 'instruction', 'answer': 'output'})


hf_dataset = load_dataset("jojo0217/korean_safe_conversation")
hf_df = hf_dataset['train'].to_pandas()[['instruction', 'output']]

hf_df = hf_df[
    (hf_df['instruction'].str.len() <= 250) & (hf_df['output'].str.len() <= 250)
]


hf_df_500 = hf_df.sample(n=500, random_state=42)

combined_df = pd.concat([filtered_df_500, hf_df_500], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
combined_df.to_csv('/mnt/hdd_6tb/bill0914/safety_finetuning/filtered_dataset3.csv', index=False)
