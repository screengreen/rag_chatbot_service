from dataclasses import dataclass

@dataclass
class TextInput:
    text: str
    language: str
    origin: str
    default: bool