# Training Guide: CORA & Qwen3-4B

## 1. CORA Benchmark Protocol
To ensure fair comparison:
1. Generate canonical train/val/test splits using identical random seeds:
   ```bash
   python ai-services/cora/scripts/split_dataset.py
   ```
2. Run TF-IDF + Linear SVM Baseline:
   ```bash
   python ai-services/cora/scripts/evaluate.py --target Product
   ```
3. Evaluate BERT (`google-bert/bert-base-uncased`) and ModernBERT (`answerdotai/ModernBERT-base`) using matching tokenization lengths and learning rates.
4. Test set remains strictly held out until the top-performing candidate is chosen.

## 2. Qwen3-4B Conversational Evaluation
1. Test pretrained base model zero-shot on banking inquiries before considering fine-tuning.
2. Fine-tuning should strictly use QLoRA (4-bit quantization with LoRA adapters) to minimize memory and GPU hours.
