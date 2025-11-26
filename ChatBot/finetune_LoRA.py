import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


from load_model import (
	finetune_model_id,
	load_lora_tokenizer,
	load_lora_model,
	load_lora_trainer
)

def lora_finetune_from_feedback(feedbacks):
	if not feedbacks:
		print("피드백 데이터가 없습니다.")
		return None

	df = pd.DataFrame(feedbacks)
	df["labels"] = df["feedback"].map({"good": 1, "bad": 0})
	df["text"] = df["answer"]

	train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=df['labels'])
	train_dataset = Dataset.from_pandas(train_df)
	test_dataset = Dataset.from_pandas(test_df)

	model_name = finetune_model_id
	tokenizer = load_lora_tokenizer(model_name)
	model = load_lora_model(model_name, num_labels=2)

	def preprocess(data):
		return tokenizer(data['text'], padding='max_length', truncation=True, max_length=64)
	train_dataset = train_dataset.map(preprocess, batched=True)
	test_dataset = test_dataset.map(preprocess, batched=True)

	trainer = load_lora_trainer(model, tokenizer, train_dataset, test_dataset)
	trainer.train()
	print("LoRA 파인튜닝 완료!")

	test_texts = ["이 제품 너무 좋아요!", "별로예요. 추천 안함."]
	inputs = tokenizer(test_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
	model.eval()
	with torch.no_grad():
		outputs = model(**inputs)
		predictions = torch.argmax(outputs.logits, dim=1)
		print("테스트 결과 :", predictions.tolist())  # 1은 긍정, 0은 부정
	return model
