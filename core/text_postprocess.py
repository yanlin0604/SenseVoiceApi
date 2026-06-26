"""
文本后处理模块
================
统一处理 ASR 原始输出，解决三类问题：
  1. SenseVoice 标签清洗：去掉 <|zh|><|NEUTRAL|><|Speech|> 等富文本标签
  2. 标点补全：可选地调用 ct-punc 标点模型，给离线/2pass 结果加标点
  3. 数字归一化(ITN)：把"二零二六年""五毫克"等口语数字转为"2026年""5mg"

所有 ASR 结果在送往前端 / Java Webhook 前都应过一遍 normalize_asr_text()。
"""
import re
from loguru import logger

# SenseVoice 富文本标签清洗。优先用官方 API，缺失时回退到正则。
try:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    _HAS_RICH = True
except Exception:  # pragma: no cover - 取决于 funasr 版本
    rich_transcription_postprocess = None
    _HAS_RICH = False
    logger.warning("未找到 funasr.rich_transcription_postprocess，将使用正则回退清洗 SenseVoice 标签。")

# 形如 <|zh|> <|NEUTRAL|> <|Speech|> <|woitn|> 的标签
_TAG_PATTERN = re.compile(r"<\|[^|]*\|>")
# 连续空白
_WS_PATTERN = re.compile(r"\s+")


def strip_rich_tags(text: str) -> str:
    """去掉 SenseVoice / FunASR 的富文本标签，返回纯文本。"""
    if not text:
        return ""
    if _HAS_RICH:
        try:
            cleaned = rich_transcription_postprocess(text)
            # 官方处理后仍可能残留少量标签，兜底再清一遍
            return _TAG_PATTERN.sub("", cleaned).strip()
        except Exception as e:
            logger.debug(f"rich_transcription_postprocess 失败，回退正则: {e}")
    cleaned = _TAG_PATTERN.sub("", text)
    # 中文文本去掉多余空格（SenseVoice 常按 token 留空格）
    cleaned = _WS_PATTERN.sub("", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# 轻量数字归一化(ITN)
# 仅覆盖中文医疗场景高频的口语数字，零依赖、CPU 友好。
# 复杂场景可换 FunASR/WeTextProcessing 的完整 ITN，这里求稳不引重依赖。
# ---------------------------------------------------------------------------
_CN_NUM = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}

# 需要把前面中文数字转阿拉伯的医疗计量单位
_UNIT_WORDS = ["毫克", "毫升", "微克", "克", "毫米", "厘米", "公斤", "千克",
               "次", "天", "周", "月", "年", "岁", "度", "小时", "分钟", "秒", "片", "粒", "支"]


def _cn_to_arabic(cn: str):
    """将一段纯中文数字串转为整数；无法解析返回 None。"""
    if not cn:
        return None
    # 纯位值串，如"二零二六"按逐位拼接(年份/编号常见)
    if all(c in _CN_NUM for c in cn):
        return int("".join(str(_CN_NUM[c]) for c in cn))
    # 带单位的常规数字，如"一百二十""三十五"
    total, section, number = 0, 0, 0
    for c in cn:
        if c in _CN_NUM:
            number = _CN_NUM[c]
        elif c in _CN_UNIT:
            unit = _CN_UNIT[c]
            if unit >= 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number if number else 1) * unit
            number = 0
        else:
            return None
    return total + section + number


_CN_NUM_SEQ = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")


def normalize_numbers(text: str) -> str:
    """把中文数字转阿拉伯数字，重点服务医疗剂量/时间/年份。"""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        seq = m.group(0)
        # 单独一个"十"等无意义片段不动（如"几十"语境），长度1且为单位时跳过
        if len(seq) == 1 and seq in _CN_UNIT:
            return seq
        val = _cn_to_arabic(seq)
        return str(val) if val is not None else seq

    return _CN_NUM_SEQ.sub(_repl, text)


def normalize_asr_text(text: str, punc_model=None) -> str:
    """
    ASR 文本统一归一化入口。
    :param text: ASR 原始输出
    :param punc_model: 可选的标点模型(AutoModel)，传入则补标点
    """
    if not text:
        return ""
    # 1. 清洗富文本标签
    text = strip_rich_tags(text)
    if not text:
        return ""
    # 2. 数字归一化
    text = normalize_numbers(text)
    # 3. 标点补全（仅对没有标点的结果有意义；标点模型自带句末标点）
    if punc_model is not None:
        try:
            punc_res = punc_model.generate(input=text)
            if punc_res and len(punc_res) > 0 and punc_res[0].get("text"):
                text = punc_res[0]["text"]
        except Exception as e:
            logger.debug(f"标点模型推理失败，返回无标点文本: {e}")
    return text.strip()
