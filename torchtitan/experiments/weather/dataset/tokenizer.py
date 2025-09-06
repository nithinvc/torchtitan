from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.experiments.weather.dataset.data_sources import WeatherSources
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


def build_weather_tokenizer(job_config: JobConfig) -> BaseTokenizer:
    return WeatherTokenizer()
