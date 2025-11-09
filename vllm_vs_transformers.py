import time
import torch
# ============================================

import # Complete Here !!!

# ============================================
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================
# Experiments Configure
# ============================================
MODEL_NAME = "NousResearch/Llama-2-7b-hf"  # Llama-2 open model
PROMPTS = ["What is the difference between supervised and unsupervised learning?",
            "Explain how gradient descent works in simple terms.",
            "What does overfitting mean in machine learning?",
            "Can you describe how convolutional neural networks process images?",
            "What is the role of activation functions in neural networks?",
            "How do word embeddings represent meaning in text?",
            "Explain what reinforcement learning is with an example.",
            "What does fine-tuning mean for large language models?",
            "How does tokenization work in natural language processing?",
            "Explain the difference between precision and recall.",
            "What is the purpose of the softmax function in classification tasks?",
            "How do transformers handle long sequences compared to RNNs?",
            "What is the idea behind self-supervised learning?",
            "Explain what a loss function does in training a model.",
            "What are the benefits of using quantization for large models?",
            "How does attention help models focus on important parts of input data?"]
TEMPERATURE=0.7
MAX_LENGTH=256
SAMPLING_PARAMS = vllm.SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_LENGTH)


def run_transformers_inference(batch_size: int):
    # Model load
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                dtype=torch.float16).to(device="cuda")
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True).to(device="cuda")
    outputs = llm.generate(**inputs, temperature=TEMPERATURE, max_length=MAX_LENGTH)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    infer_time = time.time() - start

    # throughput calculation
    tokens_generated = sum(len(o) for o in outputs)
    throughput = tokens_generated / infer_time

    print(f"Transformers Inference time: {infer_time:.2f}s")
    print(f"Transformers Throughput: {throughput:.1f} tokens/s")

    print("\nTransformers Sample output:")
    print(decoded_outputs[0][:150], "...")
    return infer_time, throughput


def run_vllm_inference(batch_size: int):
    # Model load
    start = time.time()
    llm = vllm.LLM(model=MODEL_NAME, dtype=torch.float16, gpu_memory_utilization=0.8)
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    start = time.time()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    infer_time = time.time() - start

    # throughput calculation
    tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = tokens_generated / infer_time

    print(f"vllm Inference time: {infer_time:.2f}s")
    print(f"vllm Throughput: {throughput:.1f} tokens/s")

    print("\nvllm Sample output:")
    print(outputs[0].outputs[0].text[:150], "...")
    return infer_time, throughput


if __name__ == '__main__':
    # ============================================
    # Run
    # ============================================
    transformers_results = []
    for batch_size in [8]:
        res = run_transformers_inference(batch_size=batch_size)
        transformers_results.append((batch_size, *res))
    
    torch.cuda.empty_cache()
    
    vllm_results = []
    for batch_size in [8]:
        res = run_vllm_inference(batch_size=batch_size)
        vllm_results.append((batch_size, *res))
    
    # ============================================
    # Results
    # ============================================
    print("\n=== Transformers Summary ===")
    print(f"{'Batch':<8}{'Latency(s)':<12}{'Throughput(tok/s)':<20}")
    for bsz, lat, thpt in transformers_results:
        print(f"{bsz:<8}{lat:<12.2f}{thpt:<20.1f}")
    

    print("\n=== vllm Summary ===")
    print(f"{'Batch':<8}{'Latency(s)':<12}{'Throughput(tok/s)':<20}")
    for bsz, lat, thpt in vllm_results:
        print(f"{bsz:<8}{lat:<12.2f}{thpt:<20.1f}")
