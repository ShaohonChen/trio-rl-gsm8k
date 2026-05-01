import re
import pytrio as trio
import numpy as np

# 1. 与TRIO建立连接
service_client = trio.ServiceClient()

# 2. 创建1个训练客户端
base_model = "Qwen/Qwen3-4B-Instruct-2507"
training_client = service_client.create_lora_training_client(
    base_model=base_model,
    rank=32,
)

# 3. 数据集-让LLM做简单数学题
dataset = [
    ("What is 2 + 3?", 5),
    ("What is 7 - 4?", 3),
    ("What is 6 * 8?", 48),
    ("What is 12 / 3?", 4),
    ("Solve for x: x + 5 = 9", 4),
    ("Solve for x: 2x = 10", 5),
    ("What is 3 squared?", 9),
    ("What is the square root of 81?", 9),
    ("What is 15 + 27?", 42),
    ("What is 100 - 58?", 42),
]

eval_dataset = [
    ("Solve for x: x + 7 = 12", 5),
    ("What is 9 * 7?", 63),
    ("What is 81 / 9?", 9),
    ("What is 14 + 28?", 42),
]

# 4. 获取Tokenizer
print("Loading tokenizer...")
tokenizer = training_client.get_tokenizer()
print("Tokenizer finish")

# 6. 从模型输出中解析数字答案
def parse_number(text: str):
    match = re.fullmatch(r"-?\d+(?:\.\d+)?", text.strip())
    return float(match.group()) if match else None

# 7. 奖励函数
def compute_reward(text: str, gold: float) -> float:
    pred = parse_number(text)
    if pred is None:
        return -1.0
    if abs(pred - gold) < 1e-6:
        return 2.0
    return -0.5

# 8. 转成numpy数组，方便后面统计loss
def to_np(x):
    return np.array(x.tolist() if hasattr(x, "tolist") else x, dtype=float)

# 9. 把一次采样结果处理成trio训练需要的Datum格式
def process_rollout(prompt_tokens, completion_tokens, completion_logprobs, reward_value):
    tokens = prompt_tokens + completion_tokens

    prompt_weights = [0] * len(prompt_tokens)
    completion_weights = [1] * len(completion_tokens)
    weights = prompt_weights + completion_weights

    old_logprobs = ([0.0] * len(prompt_tokens) + list(completion_logprobs))[:len(tokens)]
    old_logprobs += [0.0] * (len(tokens) - len(old_logprobs))

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
    old_logprobs = old_logprobs[1:]
    advantages = [reward_value] * (len(tokens) - 1)

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
            logprobs=old_logprobs,
            advantages=advantages,
        ),
    )

# 10. RL训练
print("Start RL Training")

for iter in range(15):
    sampler = training_client.save_weights_and_get_sampling_client(name=f"rl-math-sampler-iter{iter}")
    processed_examples = []
    rewards = []
    correct = 0
    total = 0

    for question, gold in dataset:
        prompt_tokens = tokenizer.encode(f"Question: {question}\nReturn only the final numeric answer.\nAnswer:", add_special_tokens=True)

        future_sample = sampler.sample(
            prompt=trio.ModelInput.from_ints(prompt_tokens),
            sampling_params=trio.SamplingParams(max_tokens=8, temperature=0.7),
            num_samples=4,
        )
        sample_result = future_sample.result()

        for sequence in sample_result.sequences:
            reward_value = compute_reward(sequence.text, float(gold))
            pred = parse_number(sequence.text)

            rewards.append(reward_value)
            total += 1
            correct += pred is not None and abs(pred - gold) < 1e-6

            completion_tokens = tokenizer.encode(sequence.text, add_special_tokens=False)

            if completion_tokens:
                processed_examples.append(
                    process_rollout(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        completion_logprobs=sequence.logprobs,
                        reward_value=reward_value,
                    )
                )

    print(
        f"Iter{iter+1} | Reward: {np.mean(rewards):.4f} | "
        f"Acc: {correct / max(total, 1):.4f} | Samples: {len(processed_examples)}"
    )

    fwdbwd_future = training_client.forward_backward(processed_examples, "importance_sampling")
    optim_future = training_client.optim_step(trio.AdamParams(learning_rate=1e-5))

    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    logprobs = np.concatenate([to_np(output["logprobs"]) for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([to_np(example.loss_fn_inputs["weights"]) for example in processed_examples])
    old_logprobs = np.concatenate([to_np(example.loss_fn_inputs["logprobs"]) for example in processed_examples])
    advantages = np.concatenate([to_np(example.loss_fn_inputs["advantages"]) for example in processed_examples])

    mask = weights > 0
    loss = -np.sum(np.exp(logprobs[mask] - old_logprobs[mask]) * advantages[mask]) / mask.sum()
    print(f"Iter{iter+1} IS Loss: {loss:.4f}\n")

# 11. 推理与评估
print("Start Evaluation")

sampling_base_client = service_client.create_sampling_client(base_model=base_model)
sampling_rl_client = training_client.save_weights_and_get_sampling_client(name="math-rl-final")

for question, gold in eval_dataset:
    prompt = trio.ModelInput.from_ints(
        tokenizer.encode(f"Question: {question}\nReturn only the final numeric answer.\nAnswer:", add_special_tokens=True)
    )

    future_base = sampling_base_client.sample(prompt=prompt, sampling_params=trio.SamplingParams(max_tokens=8, temperature=0.0), num_samples=1)
    future_rl = sampling_rl_client.sample(prompt=prompt, sampling_params=trio.SamplingParams(max_tokens=8, temperature=0.0), num_samples=1)
    
    result_base = future_base.result()
    result_rl = future_rl.result()
    
    base_text = result_base.sequences[0].text.strip()
    rl_text = result_rl.sequences[0].text.strip()

    print("=" * 60)
    print(f"Q: {question} | Gold: {gold}")
    print(f"Base: {repr(base_text)} -> {parse_number(base_text)}")
    print(f"RL:   {repr(rl_text)} -> {parse_number(rl_text)}")