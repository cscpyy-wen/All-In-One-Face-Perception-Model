#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese translations for expression and attribute labels.

Single source of truth for all zh-CN label mappings used across the demo
system (UI rendering, result visualization, summary images).
"""

# ---------------------------------------------------------------------------
# Expression labels (7 classes from AffectNet)
# ---------------------------------------------------------------------------
EXPRESSION_ZH: dict[str, str] = {
    "neutral": "中性",
    "happy": "高兴",
    "sad": "伤心",
    "surprise": "惊讶",
    "fear": "害怕",
    "disgust": "厌恶",
    "anger": "愤怒",
}

# ---------------------------------------------------------------------------
# Attribute labels (40 attributes from CelebA)
# ---------------------------------------------------------------------------
ATTRIBUTE_ZH: dict[str, str] = {
    "5_o_Clock_Shadow": "胡茬",
    "Arched_Eyebrows": "弓形眉",
    "Attractive": "有吸引力",
    "Bags_Under_Eyes": "眼袋",
    "Bald": "秃头",
    "Bangs": "有刘海",
    "Big_Lips": "大嘴唇",
    "Big_Nose": "大鼻子",
    "Black_Hair": "黑发",
    "Blond_Hair": "金发",
    "Blurry": "模糊",
    "Brown_Hair": "棕发",
    "Bushy_Eyebrows": "浓眉",
    "Chubby": "圆润",
    "Double_Chin": "双下巴",
    "Eyeglasses": "戴眼镜",
    "Goatee": "山羊胡",
    "Gray_Hair": "灰/白发",
    "Heavy_Makeup": "浓妆",
    "High_Cheekbones": "高颧骨",
    "Male": "男性",
    "Mouth_Slightly_Open": "嘴微张",
    "Mustache": "八字胡",
    "Narrow_Eyes": "细长眼",
    "No_Beard": "无胡须",
    "Oval_Face": "鹅蛋脸",
    "Pale_Skin": "肤色偏白",
    "Pointy_Nose": "尖鼻子",
    "Receding_Hairline": "发际线后移",
    "Rosy_Cheeks": "红脸颊",
    "Sideburns": "鬓角",
    "Smiling": "微笑",
    "Straight_Hair": "直发",
    "Wavy_Hair": "卷发",
    "Wearing_Earrings": "戴耳环",
    "Wearing_Hat": "戴帽子",
    "Wearing_Lipstick": "涂口红",
    "Wearing_Necklace": "戴项链",
    "Wearing_Necktie": "打领带",
    "Young": "年轻",
}

# ---------------------------------------------------------------------------
# Parsing labels (19 classes from CelebAMask-HQ)
# ---------------------------------------------------------------------------
PARSING_ZH: dict[str, str] = {
    "background": "背景",
    "neck": "脖子",
    "face": "脸",
    "cloth": "衣服",
    "rr": "右眉",
    "lr": "左眉",
    "rb": "右眼",
    "lb": "左眼",
    "re": "右虹膜",
    "le": "左虹膜",
    "nose": "鼻子",
    "imouth": "内嘴",
    "llip": "下唇",
    "ulip": "上唇",
    "hair": "头发",
    "glass": "眼镜",
    "hat": "帽子",
    "earr": "耳环",
    "neckl": "项链",
}
