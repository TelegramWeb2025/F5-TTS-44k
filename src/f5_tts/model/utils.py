from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import jieba
import torch
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
        text: list[str] | list[list[str]],
        vocab_char_map: dict[str, int],  # {char: idx}
        padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
                "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i: i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


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
    lr_warmup_scheduler = initial_lr * warmup_factor

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
    lr_decay_scheduler = initial_lr * decay_factor

    # 3. self.scheduler (SequentialLR) 和 self.optimizer 的学习率
    # SequentialLR 会根据 current_step 决定使用哪个子调度器
    lr_sequential_scheduler = 0
    if current_step < warmup_updates:
        # 处于预热阶段
        # SequentialLR 将 current_step (作为 last_epoch) 传递给 warmup_scheduler
        active_factor = get_linear_factor(
            current_step,
            warmup_updates,
            warmup_start_factor,
            warmup_end_factor
        )
        lr_sequential_scheduler = initial_lr * active_factor
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
        lr_sequential_scheduler = initial_lr * active_factor

    lr_optimizer = lr_sequential_scheduler

    return {
        "current_step": current_step,
        "warmup_scheduler_lr": lr_warmup_scheduler,
        "decay_scheduler_lr": lr_decay_scheduler,
        "sequential_scheduler_lr": lr_sequential_scheduler,
        "optimizer_lr": lr_optimizer,
        "info": {
            "warmup_updates": warmup_updates,
            "decay_updates": decay_updates,
            "total_steps": total_steps,
            "initial_lr": initial_lr
        }
    }
