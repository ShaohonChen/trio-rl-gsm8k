"""TRIO + GSM8K 的 importance sampling 强化学习微调教学示例。

核心流程：
1. 每个 step 先用当前 LoRA 权重创建 sampler；
2. sampler 对当前 batch 异步采样，得到 completion、logprob 和 reward；
3. 同一道题内用 reward 计算 group-relative advantage，并组装成 trio.Datum；
4. forward_backward(..., "importance_sampling") 计算梯度，再 optim_step 更新权重。
"""

import argparse
import asyncio
import math
import re

import numpy as np
import pytrio as trio
import swanlab
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRIO on-policy RL fine-tuning example for GSM8K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507", help="TRIO 中可训练的基座模型")
    parser.add_argument("--dataset-path", default="./gsm8k", help="本地 GSM8K 数据集路径")
    parser.add_argument("--dataset-config", default="main", help="datasets.load_dataset 使用的数据配置名")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=1, help="遍历训练子集的轮数")
    parser.add_argument("--train-samples", type=int, default=512, help="用于训练的 GSM8K 样本数")
    parser.add_argument("--eval-samples", type=int, default=128, help="用于最终评估的 GSM8K 样本数")
    parser.add_argument("--prompt-batch-size", type=int, default=8, help="每个 RL step 采样多少道题")
    parser.add_argument("--num-samples-per-prompt", type=int, default=4, help="每道题采样多少条回答")
    parser.add_argument("--max-tokens", type=int, default=512, help="单条回答最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="训练采样温度")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="AdamW 学习率")
    parser.add_argument("--sampling-seed", type=int, default=None, help="训练采样随机种子；None 表示不固定")
    parser.add_argument("--eval", dest="eval_model_path", default=None, help="仅评估模式，输入sample路径")
    parser.add_argument("--checkpoint-prefix", default="rl-gsm8k", help="TRIO 保存 sampler 权重时使用的前缀")
    parser.add_argument("--swanlab-project", default="GSM8K-WITH-TRIO", help="SwanLab 项目名")
    parser.add_argument("--swanlab-experiment", default="rl-gsm8k", help="SwanLab 实验名")
    return parser.parse_args()


def make_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Let's think step by step. Put your final numeric answer after '#### '.\n"
        "Answer:"
    )


def gold_answer(answer: str) -> float:
    return float(answer.split("####")[-1].strip().replace(",", ""))


def parse_model_answer(text: str) -> float | None:
    """优先解析 #### 后的最终答案；没有时退回到最后一个数字。"""
    ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
    NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
    clean = text.replace(",", "")
    match = ANSWER_RE.search(clean)
    if match:
        return float(match.group(1))
    numbers = NUMBER_RE.findall(clean)
    return float(numbers[-1]) if numbers else None


def reward_fn(text: str, gold: float) -> float:
    # 教学用规则奖励：正确答案给正奖励，答案错误或无法解析给惩罚。
    pred = parse_model_answer(text)
    if pred is None:
        return -1.0
    return 1.0 if abs(pred - gold) < 1e-6 else -0.5


def group_advantages(rewards: list[float]) -> list[float]:
    """同一道题内归一化 reward，得到 group-relative advantage。"""
    if not rewards:
        return []
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    return [(reward - mean) / (std + 1e-8) for reward in rewards]


def make_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    completion_logprobs: list[float],
    advantage: float,
) -> trio.Datum | None:
    """把一条 completion 转成 TRIO importance_sampling loss 需要的 Datum。"""
    if not completion_tokens:
        return None

    tokens = prompt_tokens + completion_tokens

    # prompt 只作为上下文，不参与 loss；completion token 才使用 advantage 训练。
    weights = ([0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens))
    advantages = [advantage * weight for weight in weights]

    # importance sampling 需要旧策略采样时的 logprob；prompt 部分补 0 并由 weights 屏蔽。
    old_logprobs = ([0.0] * len(prompt_tokens) + list(completion_logprobs))[: len(tokens)]
    old_logprobs += [0.0] * (len(tokens) - len(old_logprobs))

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "logprobs": old_logprobs[1:],
            "advantages": advantages[1:],
        },
    )


def iter_batches(dataset: list[dict], batch_size: int, epochs: int):
    for epoch in range(epochs):
        for start in range(0, len(dataset), batch_size):
            yield epoch, start, dataset[start : start + batch_size]


async def sample_one_question(sampler, tokenizer, item: dict, args: argparse.Namespace) -> dict:
    """对一道题采样多条回答，并在题目内部计算 advantage。"""
    prompt_tokens = tokenizer.encode(make_prompt(item["question"]), add_special_tokens=True)
    future = await sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        sampling_params=trio.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.sampling_seed,
        ),
        num_samples=args.num_samples_per_prompt,
    )
    sample_result = await future

    gold = gold_answer(item["answer"])
    completions = []
    completion_lens = []
    rewards=[]
    for sequence in sample_result.sequences:
        completion_tokens = list(sequence.tokens)
        reward = reward_fn(sequence.text, gold)
        pred = parse_model_answer(sequence.text)
        is_correct = pred is not None and abs(pred - gold) < 1e-6
        completions.append((completion_tokens, sequence.logprobs, is_correct))
        completion_lens.append(len(completion_tokens))
        rewards.append(reward)

    advantages = group_advantages(rewards)
    corrects = []
    datums = []
    for (completion_tokens, logprobs, is_correct), advantage in zip(completions, advantages):
        datum = make_datum(prompt_tokens, completion_tokens, logprobs, advantage)
        if datum is not None:
            datums.append(datum)
            corrects.append(is_correct)
    
    correct = sum(corrects)
    return {
        "datums": datums,
        "rewards": rewards,
        "advantages": advantages,
        "correct": correct,
        "comp_len": completion_lens,
    }


async def collect_rollouts(sampler, tokenizer, batch: list[dict], args: argparse.Namespace):
    """并发采样一个 prompt batch，返回训练 Datum 和日志指标。"""
    # batch 内每道题彼此独立，因此可以并发提交采样请求。
    results = await asyncio.gather(
        *(sample_one_question(sampler, tokenizer, item, args) for item in batch)
    )

    datums = [datum for result in results for datum in result["datums"]]
    rewards = [reward for result in results for reward in result["rewards"]]
    advantages = [adv for result in results for adv in result["advantages"]]
    correct = sum(result["correct"] for result in results)
    completion_lens = [comp_len for result in results for comp_len in result["comp_len"]]

    if not datums:
        print("No valid datums, skip this batch")
        return [], {}
    return datums, {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "advantage_std": float(np.std(advantages)),
        "accuracy": correct / len(datums),
        "completion_len_avg": float(np.mean(completion_lens)),
        "completion_len_std": float(np.std(completion_lens)),
        "batch_train_tokens": sum(completion_lens),
    }


async def train(
    training_client,
    tokenizer,
    train_dataset: list[dict],
    args: argparse.Namespace,
) -> int:
    total_steps = args.epochs * math.ceil(len(train_dataset) / args.prompt_batch_size)

    print("Start on-policy importance sampling RL training")
    for step, (epoch, batch_start, batch) in enumerate(
        iter_batches(train_dataset, args.prompt_batch_size, args.epochs)
    ):
        sampler = await training_client.save_weights_and_get_sampling_client_async(
            name=f"{args.swanlab_experiment}-step{step}"
        )
        datums, rollout_stats = await collect_rollouts(sampler, tokenizer, batch, args)
        if not datums:
            continue
        fwdbwd_future = await training_client.forward_backward_async(datums, "importance_sampling")
        optim_future = await training_client.optim_step_async(
            trio.AdamParams(learning_rate=args.learning_rate)
        )
        fwdbwd_result, _ = await asyncio.gather(fwdbwd_future, optim_future)
        loss = float(fwdbwd_result.metrics["loss:sum"]) / rollout_stats["batch_train_tokens"]

        swanlab.log({
            "train/loss": loss,
            **{f"rollout/{key}": value for key, value in rollout_stats.items()},
            "epoch": epoch,
            "batch_start": batch_start,
        }, step=step)
        print(
            f"Step {step + 1}/{total_steps} | Epoch {epoch + 1} | "
            f"Reward {rollout_stats['reward_mean']:.3f} | "
            f"Acc {rollout_stats['accuracy']:.3f} | "
            f"Batch {len(datums)} | "
            f"Loss {loss:.3f}"
        )

    return total_steps


async def evaluate(
    name: str,
    sampler,
    tokenizer,
    eval_dataset: list[dict],
    args: argparse.Namespace,
) -> dict:
    """只评估一个模型：并发采样、解析答案、统计 accuracy。"""
    print(f"Evaluating {name} model...")
    params = trio.SamplingParams(max_tokens=args.max_tokens, temperature=0.0, seed=42)

    examples = []
    sample_calls = []
    for item in eval_dataset:
        gold = gold_answer(item["answer"])
        prompt = trio.ModelInput.from_ints(
            tokenizer.encode(make_prompt(item["question"]), add_special_tokens=True)
        )
        examples.append((item["question"], gold))
        sample_calls.append(
            sampler.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
        )

    # 第一层 gather 并发提交 sample_async，第二层 gather 并发等待 TRIO 生成完成。
    sample_futures = await asyncio.gather(*sample_calls)
    sample_results = await tqdm_asyncio.gather(*sample_futures, desc="Evaluating")

    correct = 0
    for (question, gold), sample_result in zip(examples, sample_results):
        text = sample_result.sequences[0].text
        pred = parse_model_answer(text)
        is_correct = pred is not None and abs(pred - gold) < 1e-6
        correct += is_correct

        print("=" * 80)
        print(f"Model: {name}")
        print(f"Q: {question}")
        print(f"Gold: {gold}")
        print(f"Pred: {repr(text.strip())} -> {pred}")
        print(f"Correct: {is_correct}")

    total = len(eval_dataset)
    metrics = {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }
    print("=" * 80)
    print(f"{name} Accuracy: {metrics['accuracy']:.4f} ({correct}/{total})")

    return metrics


async def main():
    # 解析命令行参数
    args = parse_args()

    # 连接 TRIO 服务
    print("Connecting to TRIO service...")
    service_client = trio.ServiceClient()

    # 加载 GSM8K 数据集
    print("Loading GSM8K dataset...")
    gsm8k = load_dataset(args.dataset_path, args.dataset_config)
    eval_dataset = list(gsm8k["test"])[: args.eval_samples]

    # 仅评估模式：不训练，直接评估指定模型
    if args.eval_model_path:
        eval_sampler = await service_client.create_sampling_client_async(
            base_model=args.base_model,
            model_path=args.eval_model_path,
        )
        await evaluate("eval", eval_sampler, eval_sampler.get_tokenizer(), eval_dataset, args)
        return

    # 创建 LoRA 训练客户端
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    train_dataset = list(gsm8k["train"])[: args.train_samples]
    total_steps = args.epochs * math.ceil(len(train_dataset) / args.prompt_batch_size)

    # 初始化 SwanLab 实验追踪
    swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        config=vars(args) | {"loss_fn": "importance_sampling", "total_steps": total_steps},
    )

    # 执行强化学习训练
    total_steps = await train(training_client, tokenizer, train_dataset, args)

    # 训练完成后，评估基座模型和 RL 微调后的模型
    print("Start Evaluation on GSM8K Test Set")
    base_sampler = await service_client.create_sampling_client_async(
        base_model=args.base_model
    )
    rl_sampler = await training_client.save_weights_and_get_sampling_client_async(
        name=f"{args.checkpoint_prefix}-final"
    )
    base_metrics = await evaluate("base", base_sampler, tokenizer, eval_dataset, args)
    rl_metrics = await evaluate("rl", rl_sampler, tokenizer, eval_dataset, args)

    # 记录最终评估结果到 SwanLab
    swanlab.log({
        "eval/base_accuracy": base_metrics["accuracy"],
        "eval/rl_accuracy": rl_metrics["accuracy"],
        "eval/base_correct": base_metrics["correct"],
        "eval/rl_correct": rl_metrics["correct"],
        "eval/total": base_metrics["total"],
    }, step=total_steps)


if __name__ == "__main__":
    asyncio.run(main())
