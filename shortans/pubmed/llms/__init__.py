from .openai_model import chatgpt
from .base import LanguageModel
from .deepseek import DeepSeek

MODEL_DICT = {
    "gpt4o": "gpt-4o",
    "deepseek-r1": "deepseek-reasoner",
    "deepseek-v3": "deepseek-chat",
}


def get_model(model_name: str, **kwargs) -> LanguageModel:
    if model_name in MODEL_DICT.keys():
        model_name = MODEL_DICT[model_name]

    if "gpt" in model_name.lower():
        return chatgpt(model=model_name, **kwargs)
    elif "deepseek" in model_name.lower():
        return DeepSeek(model=model_name, **kwargs)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
