"""
文本格式化模块
包含情绪、事件字典和文本格式化函数
"""
import re


# 情绪字典
emo_dict = {
    "<|HAPPY|>": "",
    "<|SAD|>": "",
    "<|ANGRY|>": "",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "",
    "<|DISGUSTED|>": "",
    "<|SURPRISED|>": "",
}

# 事件字典
event_dict = {
    "<|BGM|>": "",
    "<|Speech|>": "",
    "<|Applause|>": "",
    "<|Laughter|>": "",
    "<|Cry|>": "",
    "<|Sneeze|>": "",
    "<|Breath|>": "",
    "<|Cough|>": "",
}

# 表情符号字典
emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "",
    "<|SAD|>": "",
    "<|ANGRY|>": "",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "",
    "<|Speech|>": "",
    "<|Applause|>": "",
    "<|Laughter|>": "",
    "<|FEARFUL|>": "",
    "<|DISGUSTED|>": "",
    "<|SURPRISED|>": "",
    "<|Cry|>": "",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "",
    "<|Breath|>": "",
    "<|Cough|>": "",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

# 语言字典
lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

# 情绪表情符号集合
emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
# 事件表情符号集合
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}


def format_str(s: str) -> str:
    """
    格式化字符串，替换特殊标记
    
    Args:
        s: 输入字符串
    
    Returns:
        格式化后的字符串
    """
    for sptk in emoji_dict:
        s = s.replace(sptk, emoji_dict[sptk])
    return s


def format_str_v2(s: str) -> str:
    """
    格式化字符串 V2版本
    处理情绪和事件标记
    
    Args:
        s: 输入字符串
    
    Returns:
        格式化后的字符串
    """
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    s = s + emo_dict[emo]

    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()


try:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess as _official_postprocess
except Exception:
    _official_postprocess = None


def format_str_v3(s: str) -> str:
    """
    格式化字符串 V3版本（集成 FunASR 官方后处理算法）
    更完善的情绪和事件处理
    
    Args:
        s: 输入字符串
    
    Returns:
        格式化后的字符串
    """
    if not s:
        return ""
    if _official_postprocess is not None:
        try:
            return _official_postprocess(s)
        except Exception:
            pass

    def get_emo(s):
        return s[-1] if (s and s[-1] in emo_set) else None
    
    def get_event(s):
        return s[0] if (s and s[0] in event_set) else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) is not None:
            s_list[i] = s_list[i][1:]
        if len(s_list[i]) == 0:
            continue
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) is not None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def clean_rich_transcription(s: str) -> str:
    """
    获取纯净的文字结果（去除 FunASR / SenseVoice 生成的所有情绪、事件 Emoji 及富标签）
    非常适用于医疗病例记录、结构化提取等严谨文本场景
    """
    text = format_str_v3(s)
    # 移除表情符号
    for emoji in emo_set.union(event_set).union({"❓"}):
        text = text.replace(emoji, "")
    # 清理多余空字符
    return re.sub(r'\s+', ' ', text).strip()


def contains_chinese_english_number(s: str) -> bool:
    """
    检查字符串是否包含中文、英文或数字
    
    Args:
        s: 输入字符串
    
    Returns:
        是否包含中文、英文或数字
    """
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s))


def remove_punctuation(text: str) -> str:
    """
    移除文本中的所有标点符号（中文和英文标点）
    用于标点符号恢复模型的输入预处理
    
    Args:
        text: 输入文本
    
    Returns:
        移除标点后的文本
    """
    # 中文标点符号
    chinese_punctuation = '！？。，、；：""''（）《》【】…—·～'
    # 英文标点符号
    english_punctuation = '!?.,;:\'"()[]{}/-_=+*&^%$#@`~|\\<>'
    
    # 合并所有标点符号
    all_punctuation = chinese_punctuation + english_punctuation
    
    # 移除所有标点符号
    for punct in all_punctuation:
        text = text.replace(punct, '')
    
    return text.strip()
