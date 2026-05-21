import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import librosa
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List
from transformers import (
    HubertForCTC,
    Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,
    TrainingArguments, Trainer
)

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR       = "path/to/uaspeech"   # update this
TRAIN_DIR      = os.path.join(BASE_DIR, "train")
TEST_DIR       = os.path.join(BASE_DIR, "test")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
TARGET_SR      = 16000
MIN_SAMPLES    = 16000

# ── Data Loading ──────────────────────────────────────────────────────
def collect_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".wav"):
                files.append(os.path.join(root, f))
    return files

transcript_lookup = {}
for f in os.listdir(TRANSCRIPT_DIR):
    if f.endswith(".txt"):
        key = f.replace(".txt", "")
        with open(os.path.join(TRANSCRIPT_DIR, f), "r") as file:
            transcript_lookup[key] = file.read().strip()

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    key      = filename.replace(".wav", "")
    speaker  = key.split("_")[0]
    word     = transcript_lookup.get(key, None)
    return {"path": filepath, "word": word, "speaker": speaker}

train_files = collect_files(TRAIN_DIR)
test_files  = collect_files(TEST_DIR)

train_df = pd.DataFrame([parse_filename(f) for f in train_files])
test_df  = pd.DataFrame([parse_filename(f) for f in test_files])
train_df = train_df.dropna(subset=["word"]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["word"]).reset_index(drop=True)

# ── Preprocessing ─────────────────────────────────────────────────────
def load_and_preprocess(filepath):
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = T.Resample(orig_freq=sr, new_freq=TARGET_SR)(waveform)
    waveform = waveform.squeeze().numpy()
    waveform, _ = librosa.effects.trim(waveform, top_db=30, frame_length=512, hop_length=128)
    return waveform

def get_trimmed_length(filepath):
    return len(load_and_preprocess(filepath))

train_df = train_df[train_df["path"].apply(lambda p: get_trimmed_length(p) >= MIN_SAMPLES)].reset_index(drop=True)
test_df  = test_df[test_df["path"].apply(lambda p: get_trimmed_length(p) >= MIN_SAMPLES)].reset_index(drop=True)

# ── Vocabulary ────────────────────────────────────────────────────────
all_words = train_df["word"].tolist() + test_df["word"].tolist()
all_chars = sorted(set("".join(all_words)))
vocab = {char: idx for idx, char in enumerate(all_chars)}
vocab["[PAD]"] = len(vocab)
vocab["[UNK]"] = len(vocab)
vocab["|"]     = len(vocab)

os.makedirs("model", exist_ok=True)
with open("model/vocab.json", "w") as f:
    json.dump(vocab, f)

# ── Processor ─────────────────────────────────────────────────────────
tokenizer = Wav2Vec2CTCTokenizer(
    "model/vocab.json", unk_token="[UNK]",
    pad_token="[PAD]", word_delimiter_token="|"
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000,
    padding_value=0.0, do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# ── Dataset ───────────────────────────────────────────────────────────
class UASpeechDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df        = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        audio  = load_and_preprocess(row["path"])
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
        labels = self.processor.tokenizer(row["word"]).input_ids
        return {"input_values": inputs.input_values.squeeze(), "labels": torch.tensor(labels)}

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding:   bool = True

    def __call__(self, features: List[Dict]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids":    f["labels"]}       for f in features]
        batch        = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

train_dataset = UASpeechDataset(train_df, processor)
test_dataset  = UASpeechDataset(test_df,  processor)
data_collator = DataCollatorCTCWithPadding(processor=processor)

# ── Model ─────────────────────────────────────────────────────────────
model = HubertForCTC.from_pretrained("facebook/hubert-base-ls960", ignore_mismatched_sizes=True)
model.lm_head = nn.Linear(768, len(processor.tokenizer), bias=True)
nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
nn.init.zeros_(model.lm_head.bias)
model.config.vocab_size        = len(processor.tokenizer)
model.config.mask_time_prob    = 0.0
model.config.mask_feature_prob = 0.0
for param in model.parameters():
    param.requires_grad = True
model = model.to("cuda")

# ── Training ──────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="model/checkpoints",
    group_by_length=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    eval_strategy="no",
    num_train_epochs=10,
    fp16=True,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.005,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
model.save_pretrained("model/final")
processor.save_pretrained("model/final")
print("Training complete!")
