from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
import re
import warnings
from typing import List
import logging
import yaml
 
import torch

#TODO: move to device
#TODO: add hf token 




# Чтение YAML-конфигурации из файла
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)



class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]
 
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


model_name = "chavinlo/alpaca-native"  # Example model name
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
generation_config = GenerationConfig(**config)

class LLMLoader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name,  generation_config):
        if not self._initialized:
            self.model_name = model_name
            self.model = None
            self.stop_tokens = [["Human", ":"], ["AI", ":"]]
            self.tokenizer = None
            self._initialized = True
            self.stopping_criteria = None
            self.generation_config = generation_config

    def load_model(self):
        if self.model is None:
            logging.info(f"llm name: {self.model_name}")
            # self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        return self.model

    def load_tokenizer(self):
        if self.tokenizer is None:
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        return self.tokenizer
    
    def load_generation_pipeline(self):
        self.llm = pipeline(
            model=self.load_model(),
            tokenizer=self.load_tokenizer(),
            return_full_text=True,
            task="text-generation",
            stopping_criteria=StoppingCriteriaList([StopGenerationCriteria(self.stop_tokens, self.tokenizer, self.model.device)]),
            generation_config=self.generation_config, 
        )
        
    
    def get_llm(self):
        return self.llm
    

llmloader = LLMLoader(model_name, generation_config)


