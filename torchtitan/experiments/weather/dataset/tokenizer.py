from torchtitan.components.tokenizer import BaseTokenizer
from typing_extensions import override


class WeatherTokenizer(BaseTokenizer):
    def __init__(self):
        pass

    @override
    def encode(self, *args, **kwargs) -> list[int]:
        return [0]

    @override
    def decode(self, *args, **kwargs) -> str:
        return ""

    @override
    def get_vocab_size(self) -> int:
        return 1
