"""Vendored LingBot-Vision backbone implementation.

Source: https://github.com/Robbyant/lingbot-vision
Revision: 151e46321bae4399f8568829f190c7bdec216b49
License: Apache-2.0
"""

from rfdetr.models.backbone._lingbot_vision.vit import LingBotVisionTransformer, vit_small

__all__ = ["LingBotVisionTransformer", "vit_small"]
