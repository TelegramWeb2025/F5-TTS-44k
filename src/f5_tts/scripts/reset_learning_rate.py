import shutil

import torch
import argparse


def get_linear_factor(current_iter_in_phase, total_iters_in_phase, start_factor, end_factor):
    """
    计算 LinearLR 在当前阶段迭代次数下的学习率因子。
    current_iter_in_phase: 当前在当前阶段内的迭代次数 (0-indexed)。
    total_iters_in_phase: 当前阶段的总迭代次数。
    start_factor: 初始因子。
    end_factor: 最终因子。
    """
    if total_iters_in_phase <= 0:
        # 如果阶段没有迭代次数，则直接使用最终因子（例如，阶段被跳过或已完成）
        return end_factor

    # 确保迭代次数在 [0, total_iters_in_phase] 范围内，以计算进度
    # PyTorch LinearLR 的 last_epoch 从 0 到 total_iters
    # 当 last_epoch == 0, 进度为 0, 因子为 start_factor
    # 当 last_epoch == total_iters, 进度为 1, 因子为 end_factor
    progress = min(1.0, max(0.0, float(current_iter_in_phase) / total_iters_in_phase))

    return start_factor + (end_factor - start_factor) * progress


def calculate_learning_rates(current_step, total_steps, initial_lr, warmup_updates):
    """
    计算在 current_step 时各个调度器和优化器的学习率。

    参数:
    current_step (int): 当前的训练步数 (0-indexed).
    total_steps (int): 总的训练步数.
    initial_lr (float): 优化器设置的初始学习率.
    warmup_updates (int): 预热阶段的步数.

    返回:
    dict: 包含各个组件学习率的字典。
    """

    warmup_start_factor = 1e-8
    warmup_end_factor = 1.0

    decay_start_factor = 1.0
    decay_end_factor = 1e-8

    decay_updates = total_steps - warmup_updates
    if decay_updates < 0:
        # 如果总步数小于预热步数，衰减阶段实际上不存在或为0
        # 为了计算一致性，设为0，get_linear_factor会处理total_iters_in_phase <= 0的情况
        decay_updates = 0

        # 1. warmup_scheduler 的学习率
    # warmup_scheduler 的内部迭代计数器 (last_epoch)
    # 如果 current_step < warmup_updates, 它的 last_epoch 是 current_step
    # 如果 current_step >= warmup_updates, 它已经完成了 warmup_updates 次迭代, last_epoch 是 warmup_updates
    internal_step_for_warmup = min(current_step, warmup_updates)
    warmup_factor = get_linear_factor(
        internal_step_for_warmup,
        warmup_updates,
        warmup_start_factor,
        warmup_end_factor
    )
    warmup_scheduler_lr = initial_lr * warmup_factor

    # 2. decay_scheduler 的学习率
    # decay_scheduler 的内部迭代计数器 (last_epoch)
    # 如果 current_step < warmup_updates, SequentialLR 还没有调用过 decay_scheduler.step(),
    #   所以它的 last_epoch 仍是初始值 (通常是0，如果是第一次被查询)。
    # 如果 current_step >= warmup_updates, SequentialLR 会调用 decay_scheduler.step(),
    #   其 last_epoch 是 (current_step - warmup_updates)。
    if current_step < warmup_updates:
        internal_step_for_decay = 0
    else:
        internal_step_for_decay = current_step - warmup_updates

    # 确保 internal_step_for_decay 不超过 decay_updates
    internal_step_for_decay_capped = min(internal_step_for_decay, decay_updates)

    decay_factor = get_linear_factor(
        internal_step_for_decay_capped,
        decay_updates,
        decay_start_factor,
        decay_end_factor
    )
    decay_scheduler_lr = initial_lr * decay_factor

    # 3. self.scheduler (SequentialLR) 和 self.optimizer 的学习率
    # SequentialLR 会根据 current_step 决定使用哪个子调度器
    sequential_scheduler_lr = 0
    if current_step < warmup_updates:
        # 处于预热阶段
        # SequentialLR 将 current_step (作为 last_epoch) 传递给 warmup_scheduler
        active_factor = get_linear_factor(
            current_step,
            warmup_updates,
            warmup_start_factor,
            warmup_end_factor
        )
        sequential_scheduler_lr = initial_lr * active_factor
    else:
        # 处于衰减阶段
        # SequentialLR 将 (current_step - warmup_updates) (作为 last_epoch) 传递给 decay_scheduler
        step_in_decay_phase = current_step - warmup_updates
        active_factor = get_linear_factor(
            step_in_decay_phase,
            decay_updates,
            decay_start_factor,
            decay_end_factor
        )
        sequential_scheduler_lr = initial_lr * active_factor

    optimizer_lr = sequential_scheduler_lr

    return {
        "current_step": current_step,
        "warmup_scheduler_lr": warmup_scheduler_lr,
        "decay_scheduler_lr": decay_scheduler_lr,
        "sequential_scheduler_lr": sequential_scheduler_lr,
        "optimizer_lr": optimizer_lr,
        "info": {
            "warmup_updates": warmup_updates,
            "decay_updates": decay_updates,
            "total_steps": total_steps,
            "initial_lr": initial_lr
        }
    }


def update_checkpoint_lr(model_file_path, total_steps):
    """
    加载 checkpoint，修改学习率，并覆盖原文件。

    Args:
        model_file_path (str): checkpoint 文件路径
        total_steps (float): 新的训练步数
    """
    try:
        # 加载 checkpoint
        checkpoint = torch.load(model_file_path, weights_only=True, map_location="cpu")
        warmup_steps = checkpoint['scheduler_state_dict']['_milestones'][0]
        expect_lrs = calculate_learning_rates(checkpoint['scheduler_state_dict']['last_epoch'], total_steps,
                                              checkpoint['optimizer_state_dict']['param_groups'][0]['initial_lr'],
                                              warmup_steps)
        # 打印原始调度器状态
        print("原始调度器:", checkpoint['scheduler_state_dict'])

        # 修改调度器状态中的学习率
        checkpoint['scheduler_state_dict']['_last_lr'] = [expect_lrs['sequential_scheduler_lr']]
        checkpoint['scheduler_state_dict']['_schedulers'][1]['_last_lr'] = [expect_lrs['decay_scheduler_lr']]
        checkpoint['scheduler_state_dict']['_schedulers'][1]['total_iters'] = total_steps - warmup_steps

        # 打印原始优化器参数组
        print("原始优化器:", checkpoint['optimizer_state_dict']['param_groups'])

        # 修改优化器学习率
        checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = expect_lrs['optimizer_lr']
        shutil.copy(model_file_path, model_file_path + ".bak")
        # 覆盖原文件
        torch.save(checkpoint, model_file_path)

        # 验证保存结果
        reloaded_checkpoint = torch.load(model_file_path, weights_only=True, map_location="cpu")
        print("修改后的调度器:", reloaded_checkpoint['scheduler_state_dict'])
        print("修改后的优化器:", reloaded_checkpoint['optimizer_state_dict']['param_groups'])

    except FileNotFoundError:
        print(f"错误: 文件 {model_file_path} 不存在")
    except Exception as e:
        print(f"处理 checkpoint 失败: {e}")


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="修改 checkpoint 文件中的学习率并覆盖原文件")
    parser.add_argument("--model_file", type=str, required=True, help="checkpoint 文件路径（如 ckpts/model_50000.pt）")
    parser.add_argument("--total_steps", type=int, required=True, help="总步数")

    # 解析参数
    args = parser.parse_args()

    # 调用更新函数
    update_checkpoint_lr(args.model_file, args.total_steps)


if __name__ == "__main__":
    main()
