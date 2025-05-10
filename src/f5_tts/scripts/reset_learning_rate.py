import torch
import argparse


def update_checkpoint_lr(model_file_path, new_lr):
    """
    加载 checkpoint，修改学习率，并覆盖原文件。

    Args:
        model_file_path (str): checkpoint 文件路径
        new_lr (float): 新的学习率
    """
    try:
        # 加载 checkpoint
        checkpoint = torch.load(model_file_path, weights_only=True, map_location="cpu")

        # 打印原始调度器状态
        print("原始调度器状态:", checkpoint['scheduler_state_dict'])

        # 修改调度器状态中的学习率
        checkpoint['scheduler_state_dict']['_last_lr'] = [new_lr]
        checkpoint['scheduler_state_dict']['_schedulers'][0]['_last_lr'] = [new_lr]
        checkpoint['scheduler_state_dict']['_schedulers'][1]['_last_lr'] = [new_lr]

        # 打印原始优化器参数组
        print("原始优化器参数组:", checkpoint['optimizer_state_dict']['param_groups'])

        # 修改优化器学习率
        checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = new_lr

        # 覆盖原文件
        torch.save(checkpoint, model_file_path)
        print(f"修改后的 checkpoint 已覆盖保存至: {model_file_path}")

        # 验证保存结果
        reloaded_checkpoint = torch.load(model_file_path, weights_only=True, map_location="cpu")
        print("验证 - 修改后的调度器学习率:", reloaded_checkpoint['scheduler_state_dict']['_last_lr'])
        print("验证 - 修改后的优化器学习率:", reloaded_checkpoint['optimizer_state_dict']['param_groups'][0]['lr'])

    except FileNotFoundError:
        print(f"错误: 文件 {model_file_path} 不存在")
    except Exception as e:
        print(f"处理 checkpoint 失败: {e}")


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="修改 checkpoint 文件中的学习率并覆盖原文件")
    parser.add_argument("--model_file", type=str, required=True, help="checkpoint 文件路径（如 ckpts/model_50000.pt）")
    parser.add_argument("--lr", type=float, required=True, help="新的学习率（如 7.5e-5）")

    # 解析参数
    args = parser.parse_args()

    # 调用更新函数
    update_checkpoint_lr(args.model_file, args.lr)


if __name__ == "__main__":
    main()