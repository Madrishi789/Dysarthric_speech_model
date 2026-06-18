# Dysarthric Speech Recognition with HuBERT

Fine-tuned HuBERT Base model for dysarthric speech recognition using the UASpeech dataset.

## Model
Hosted on Hugging Face: [Madrishi789/dysarthric-hubert](https://huggingface.co/Madrishi789/dysarthric-hubert)

## Dataset
UASpeech — speech from dysarthric speakers with cerebral palsy.

## Training
- Base model: facebook/hubert-base-ls960
- Fine-tuned with CTC loss
- 30 epochs, UASpeech train/test split
- Feature extractor CNN frozen, encoder fine-tuned
- Silence trimming with librosa
- Learning rate: 3e-5 with cosine schedule

## Usage
1. Update `BASE_DIR` in `train.py` to point to your UASpeech data
2. Run: `python train.py`
3. Model saved to `model/final/`
