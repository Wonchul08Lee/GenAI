
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

embedding_model_id = "jhgan/ko-sbert-sts"
qa_model_id = "monologg/koelectra-base-v3-finetuned-korquad"
gen_model_id = "beomi/gemma-ko-2b"
finetune_model_id = "beomi/kcbert-base"

tokenizer_opt = {
    "max_length": 512,
    "stride": 32,
    "truncation": True,
    "do_sample": True,
    "device": -1
}

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=embedding_model_id)

@st.cache_resource
def load_QA_model(tokenizer_opt):
    from transformers import pipeline
    safe_opts = {"truncation": True, "max_length": 512, "stride": 128, "padding": "max_length"}
    qa_pipeline = pipeline(
        task="question-answering",
        model=qa_model_id,
        tokenizer_kwargs=safe_opts,
        max_length=512,
        stride=128,
        device=-1
    )
    return qa_pipeline

@st.cache_resource
def load_Gen_model():
    tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
    model = AutoModelForCausalLM.from_pretrained(gen_model_id)
    return tokenizer, model

def load_lora_tokenizer(model_name=finetune_model_id):
    return AutoTokenizer.from_pretrained(model_name)

@st.cache_resource
def load_lora_model(model_name=finetune_model_id, num_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"],
    )
    return get_peft_model(model, peft_config)

def load_lora_trainer(model, tokenizer, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir="./saved_models/peft_lora_feedback",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir="./logs",
        logging_steps=10,
        label_names=["labels"],
        use_cpu=True,
    )
    import numpy as np
    def compute_metrics(predict):
        preds = np.argmax(predict.predictions, axis=1)
        acc = (preds == predict.label_ids).mean()
        return {"accuracy": acc}
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
