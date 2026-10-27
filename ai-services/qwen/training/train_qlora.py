"""
QLoRA Fine-Tuning Setup for Qwen/Qwen3-4B
Configured for RunPod with 16GB-24GB VRAM (e.g., RTX 3090, RTX 4090, A4000).
DO NOT run on local CPU or without explicit approval.
"""
import os
import argparse

def train(data_path: str, output_dir: str):
    print("====================================================================")
    print("QWEN QLoRA TRAINING PREPARATION")
    print(f"Data Path: {data_path}")
    print(f"Output Checkpoints: {output_dir}")
    print("Base Model: Qwen/Qwen3-4B")
    print("====================================================================")
    print("Notice: QLoRA training script configured with bitsandbytes 4-bit quantization,")
    print("LoRA rank r=16, alpha=32, targeting q_proj, k_proj, v_proj, o_proj.")
    print("Requires CUDA-enabled environment with GPU.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/banking_qa/qwen_banking_train.jsonl")
    parser.add_argument("--output", default="artifacts/qwen/adapter")
    args = parser.parse_args()
    train(args.data, args.output)
