# PEFT_for_Text_Summarization
This project demonstrates how to apply **PEFT (Parameter-Efficient Fine-Tuning)** using **LoRA (Low-Rank Adaptation)** for text summarization tasks. We use the **BART model** (a pre-trained sequence-to-sequence model) from the Hugging Face Transformers library and fine-tune it using LoRA to improve the efficiency of training while achieving strong performance in summarization.

## Key Components
- **PEFT**: A method to fine-tune pre-trained models with fewer parameters, enabling faster training with less computational cost.
- **LoRA**: A technique used within PEFT to adjust model weights in a low-rank manner, which helps achieve parameter efficiency during fine-tuning.
- **BART**: A sequence-to-sequence transformer model pre-trained on text generation tasks like summarization, which is used as the backbone for this project.

## Project Overview
The project uses the facebook/bart-large-cnn model for text summarization and fine-tunes it using PEFT with LoRA on the CNN/DailyMail dataset.
The following key tasks are covered in this project:
- **Loading Pre-trained Model**: We start by loading a pre-trained BART model and its tokenizer from the Hugging Face Model Hub.
- **PEFT Configuration**: We apply the LoRA configuration on the BART model to make the fine-tuning process more parameter-efficient.
- **Data Preprocessing**: The CNN/DailyMail dataset is preprocessed and tokenized.
- **Training**: The model is fine-tuned on the preprocessed dataset using the PEFT approach.
- **Summarization**: We test the fine-tuned model by summarizing a sample text.

---

## Requirements
To run the code, you need to have the following libraries installed:
```bash
pip install torch transformers datasets peft
```
### Dataset Link: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

---

## Future Work
- **Model Tuning**: Further exploration of different PEFT techniques and LoRA configurations could lead to better efficiency and model performance.
- **Deployment**: Once fine-tuned, the model can be deployed for real-time summarization tasks.
