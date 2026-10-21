"""
text_formatter 语气助词判定单元测试

对应 2pass 定稿过滤需求：ASR 单独识别出的「嗯」「哦」「呃。」等片段
应整段丢弃，不应污染转写面板、webhook 推送与下游字段提取。

运行：
    cd medical-asr-service && python -m pytest tests/test_text_formatter.py -q
"""
import sys
from pathlib import Path

import pytest

# pytest 的 rootdir 是 tests/，需显式把服务根目录加入 sys.path 才能 import modules.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.text_formatter import is_filler_only  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "嗯",              # 最基本语气词
        "哦",
        "呃",
        "啊",
        "唔",
        "诶",
        "嗯。",            # 标点恢复模型输出
        "哦？",
        "呃……",
        "哦、呃",          # 多个语气词拼接
        "嗯哦",
        "嗯嗯",            # 重复
        " 嗯 ",            # 首尾空白
        "。",              # 纯标点，同样视为无意义片段
        "，。？！",
        "嗯．",            # 全角句点 U+FF0E（回归保护：手写字符集时曾漏网）
        "嗯＿",            # 全角下划线 U+FF3F
        "哦\t",            # 制表符
        "嗯\u200b",        # 零宽空格
    ],
)
def test_drops_filler_only_segments(text):
    """整段仅语气助词/标点时，判定为可丢弃。"""
    assert is_filler_only(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "嗯，血压120",
        "哦对，主诉头晕3天",
        "呃，既往有高血压病史",
        "哎呀",            # 回归保护：可能携带疼痛/情绪线索，不得误丢
        "哎哟",
        "哎呀，疼",
        "哈哈",            # 回归保护：笑声不作为语气词丢弃
        "呀",              # 未收录的易混字，保持保守行为
        "患者",
        "123",
    ],
)
def test_keeps_meaningful_segments(text):
    """含实际语义的片段不得被丢弃。"""
    assert is_filler_only(text) is False


def test_empty_text_is_not_treated_as_filler():
    """空串由调用方处理，本函数不负责判定，避免误丢空消息。"""
    assert is_filler_only("") is False
