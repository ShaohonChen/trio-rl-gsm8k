import re
import pytrio as trio
import numpy as np
from datasets import load_dataset
import asyncio
import swanlab


async def main():
    # 连接 TRIO 服务
    service_client = trio.ServiceClient()

    # 创建 LoRA 训练客户端
    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=32,
    )

    # 载入数据集
    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("./gsm8k", "main")
    print("Loading finish")
    
    # 获取 Tokenizer
    print("Loading tokenizer...")
    tokenizer = training_client.get_tokenizer()
    print("Tokenizer finish")
    
    # 为了演示训练效率，这里切片取子集，实际训练可以去掉切片
    train_dataset = list(gsm8k["train"])[:512]
    eval_dataset = list(gsm8k["test"])[:128]

    # 初始化 swanlab
    swanlab.init(
        project="gsm8k",
        experiment_name="dataset-512",
        config={
            "base_model": base_model,
            "lora_rank": 32,
            "num_iterations": 15,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "num_samples_per_prompt": 4,
            "max_tokens": 512,
            "temperature": 0.7,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "loss_fn": "importance_sampling",
            "epoch": 2,
        },
    )

    def extract_gold_answer(answer_str: str) -> float:
        """从 GSM8K 的原始答案中提取纯数字金标"""
        # GSM8K 答案格式通常是: "推理过程... #### 1234"
        ans = answer_str.split("####")[-1].strip()
        return float(ans.replace(",", ""))

    def parse_model_answer(text: str):
        """从模型的 CoT 输出中解析最终数字"""
        # 优先匹配我们 prompt 中要求输出的格式 #### 数字
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text.replace(",", ""))
        if match:
            return float(match.group(1))
        
        # 备选回退方案：直接找文本里的最后一个数字
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(",", ""))
        if numbers:
            return float(numbers[-1])
            
        return None

    def compute_reward(text: str, gold: float) -> float:
        pred = parse_model_answer(text)
        if pred is None:  # 格式错误或未输出数字
            return -1.0      
        if abs(pred - gold) < 1e-6:  # 答案完全正确
            return 2.0       
        return -0.5  # 答案错误

    def to_np(x):
        return np.array(x.tolist() if hasattr(x, "tolist") else x, dtype=float)

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

    def make_prompt(question: str) -> str:
        """构建思维链 Prompt"""
        return f"Question: {question}\nLet's think step by step. Put your final numeric answer after '#### '.\nAnswer:"

    print("Start RL Training on GSM8K")

    for iter_idx in range(swanlab.config["epoch"]):
        sampler = await training_client.save_weights_and_get_sampling_client_async(name=f"rl-gsm8k-sampler-iter{iter_idx}")
        processed_examples = []
        rewards = []
        correct = 0
        total = 0

        text_lengths = []
        token_lengths = []

        async def get_processed_examples(future_sample, gold, prompt_tokens, batch_start):
            nonlocal correct, total
            sample_result = await future_sample
            print(f"epoch{iter_idx} batch_start:{batch_start} sample finish")

            for sequence in sample_result.sequences:
                reward_value = compute_reward(sequence.text, gold)
                pred = parse_model_answer(sequence.text)

                rewards.append(reward_value)
                total += 1
                correct += (pred is not None and abs(pred - gold) < 1e-6)

                text_lengths.append(len(sequence.text))
                completion_tokens = tokenizer.encode(sequence.text, add_special_tokens=False)
                token_lengths.append(len(completion_tokens))

                if completion_tokens:
                    processed_examples.append(
                        process_rollout(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            completion_logprobs=sequence.logprobs,
                            reward_value=reward_value,
                        )
                    )

        # 开始进行Rollout，每批异步sample 8次，避免产生过大的并发
        print("Start Rollout")
        
        sample_batch_size = 8
        for batch_start in range(0, len(train_dataset), sample_batch_size):
            batch_items = train_dataset[batch_start:batch_start + sample_batch_size]
            sample_queue = []

            for item in batch_items:
                question = item["question"]
                gold = extract_gold_answer(item["answer"])

                prompt_str = make_prompt(question)
                prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=True)

                future_sample = await sampler.sample_async(
                    prompt=trio.ModelInput.from_ints(prompt_tokens),
                    sampling_params=trio.SamplingParams(max_tokens=swanlab.config["max_tokens"], temperature=swanlab.config["temperature"], seed=42),
                    num_samples=4,
                )

                sample_queue.append(get_processed_examples(future_sample, gold, prompt_tokens, batch_start))

            await asyncio.gather(*sample_queue)

        print(
            f"Iter{iter_idx+1} | Reward: {np.mean(rewards):.4f} | "
            f"Acc: {correct / max(total, 1):.4f} | Samples: {len(processed_examples)}"
        )

        if not processed_examples:
            print("No valid samples generated, skipping optimization step.")
            continue
        
        # 执行 Importance Sampling 前后向传播与优化（分 batch，每批 32 条）
        batch_size = swanlab.config["batch_size"]
        all_loss_fn_outputs = []
        iter_losses = []
        for batch_start in range(0, len(processed_examples), batch_size):
            batch = processed_examples[batch_start: batch_start + batch_size]
            fwdbwd_future = training_client.forward_backward(batch, "importance_sampling")
            optim_future = training_client.optim_step(trio.AdamParams(learning_rate=swanlab.config["learning_rate"]))

            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()
            all_loss_fn_outputs.extend(fwdbwd_result.loss_fn_outputs)

            logprobs = np.concatenate([to_np(output["logprobs"]) for output in fwdbwd_result.loss_fn_outputs])
            weights = np.concatenate([to_np(example.loss_fn_inputs["weights"]) for example in batch])
            old_logprobs = np.concatenate([to_np(example.loss_fn_inputs["logprobs"]) for example in batch])
            advantages = np.concatenate([to_np(example.loss_fn_inputs["advantages"]) for example in batch])

            mask = weights > 0
            loss = -np.sum(np.exp(logprobs[mask] - old_logprobs[mask]) * advantages[mask]) / mask.sum()
            iter_losses.append(loss)
            print(f"Iter{iter_idx+1} Batch{batch_start}~{batch_start + batch_size} IS Loss: {loss:.4f}")
            swanlab.log({"Batch/IS Loss": loss})

        log_info = {
            "reward/reward_mean": np.mean(rewards),
            "reward/reward_std": np.std(rewards),
            "accuracy": correct / max(total, 1),
            "valid_samples": len(processed_examples),
            "total_rollouts": total,
            "length/text_length_mean": np.mean(text_lengths) if text_lengths else 0,
            "length/text_length_max": np.max(text_lengths) if text_lengths else 0,
            "length/token_length_mean": np.mean(token_lengths) if token_lengths else 0,
            "length/token_length_max": np.max(token_lengths) if token_lengths else 0,
            "train/is_loss": np.mean(iter_losses) if iter_losses else 0,
        }
        
        swanlab.log(log_info, step=iter_idx)

        print(log_info)

    print("Start Evaluation on GSM8K Test Set")

    sampling_base_client = service_client.create_sampling_client(base_model=base_model)
    sampling_rl_client = await training_client.save_weights_and_get_sampling_client_async(name="gsm8k-rl-final")

    eval_base_correct = 0
    eval_rl_correct = 0
    eval_total = 0

    async def eval_one(item, gold, future_base, future_rl):
        nonlocal eval_base_correct, eval_rl_correct, eval_total

        result_base, result_rl = await asyncio.gather(future_base, future_rl)

        base_text = result_base.sequences[0].text.strip()
        rl_text = result_rl.sequences[0].text.strip()

        base_pred = parse_model_answer(base_text)
        rl_pred = parse_model_answer(rl_text)

        eval_total += 1
        eval_base_correct += (base_pred is not None and abs(base_pred - gold) < 1e-6)
        eval_rl_correct += (rl_pred is not None and abs(rl_pred - gold) < 1e-6)

        print("=" * 80)
        print(f"Q: {item['question']}")
        print(f"Gold Answer: {gold}")
        print("-" * 40)
        print(f"[Base Model Prediction] -> {base_pred}")
        print("-" * 40)
        print(f"[RL Model Prediction]   -> {rl_pred}")

    # 评估阶段，每批异步 sample，与 rollout 阶段保持一致
    eval_batch_size = 8
    for batch_start in range(0, len(eval_dataset), eval_batch_size):
        batch_items = eval_dataset[batch_start:batch_start + eval_batch_size]
        eval_tasks = []

        for item in batch_items:
            question = item["question"]
            gold = extract_gold_answer(item["answer"])

            prompt_str = make_prompt(question)
            prompt = trio.ModelInput.from_ints(tokenizer.encode(prompt_str, add_special_tokens=True))
            eval_params = trio.SamplingParams(max_tokens=swanlab.config["max_tokens"], temperature=0.0, seed=42)

            future_base = await sampling_base_client.sample_async(prompt=prompt, sampling_params=eval_params, num_samples=1)
            future_rl = await sampling_rl_client.sample_async(prompt=prompt, sampling_params=eval_params, num_samples=1)

            eval_tasks.append(eval_one(item, gold, future_base, future_rl))

        await asyncio.gather(*eval_tasks)

    print("=" * 80)
    print(f"Evaluation Results ({eval_total} samples):")
    print(f"  Base Model Accuracy: {eval_base_correct / max(eval_total, 1):.4f} ({eval_base_correct}/{eval_total})")
    print(f"  RL   Model Accuracy: {eval_rl_correct / max(eval_total, 1):.4f} ({eval_rl_correct}/{eval_total})")
    
    swanlab.log({
        "eval/base_accuracy": eval_base_correct / max(eval_total, 1),
        "eval/rl_accuracy": eval_rl_correct / max(eval_total, 1),
    })


if __name__ == "__main__":
    asyncio.run(main())