# Dysarthric Speech Recognition with HuBERT

Fine-tuned HuBERT Base model for dysarthric speech recognition using the UASpeech dataset.

## Model
Hosted on Hugging Face: [Madrishi789/dysarthric-hubert](https://huggingface.co/Madrishi789/dysarthric-hubert)

## Dataset
UASpeech — речь от дисартрических говорящих с церебральным параличом.

## Training
- Base model: facebook/hubert-base-ls960
- Fine-tuned with CTC loss
- 10 epochs, UASpeech train/test split
- Silence trimming with librosa
