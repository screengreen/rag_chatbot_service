from dataclasses import dataclass
from typing import Optional

@dataclass
class TextInput:
    text: str
    language: Optional[str] = None
    origin: Optional[str] = None
    default: Optional[bool] = None
    document: Optional[str] = None