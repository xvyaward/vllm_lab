import time
import torch
from vllm import LLM, SamplingParams

# ============================================
# Experiments Configure
# ============================================
MODEL_NAME = "NousResearch/Llama-2-7b-hf"       # Llama-2 open model
# MODEL_NAME = "NousResearch/Meta-Llama-3-8B"            # Llama-3 open model

PROMPT = "Explain the concept of attention mechanism in transformers."

SAMPLING_PARAMS = SamplingParams(temperature=0.7, max_tokens=256)
GPU_MEMORY_UTILIZATION = 0.8

def run_inference(use_paged_attention: bool, batch_size: int):
    """
        PagedAttention ON: block size 16
        PagedAttention OFF simulated : block size 2048
    """

    block_size = 16 if use_paged_attention else 2048

    # Model load
    start = time.time()
    llm = LLM(model=MODEL_NAME, dtype=torch.float16, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, block_size=block_size)
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    # offline batched inference with multiple prompts
    prompts = [PROMPT] * batch_size

    start = time.time()
    outputs = llm.generate(prompts, SAMPLING_PARAMS)
    infer_time = time.time() - start

    # throughput calculation
    tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = tokens_generated / infer_time

    print(f"Inference time: {infer_time:.2f}s")
    print(f"Throughput: {throughput:.1f} tokens/s")

    print("\nSample output:")
    print(outputs[0].outputs[0].text[:150], "...")
    return infer_time, throughput


if __name__ == '__main__':
    # ============================================
    # Run
    # ============================================
    results = []
    for batch_size in [8]:
        for paged in [True, False]:
            res = run_inference(use_paged_attention=paged, batch_size=batch_size)
            results.append((paged, batch_size, *res))

    # ============================================
    # Results
    # ============================================
    print("\n=== Summary ===")
    print(f"{'PagedAttn':<12}{'Batch':<8}{'Latency(s)':<12}{'Throughput(tok/s)':<20}")
    for paged, bsz, lat, thpt in results:
        print(f"{'ON' if paged else 'OFF':<12}{bsz:<8}{lat:<12.2f}{thpt:<20.1f}")
