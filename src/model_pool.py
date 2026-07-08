# src/model_pool.py
import os
from langchain_groq import ChatGroq

POOLS = [
    [os.getenv("GROQ_KEY1"), os.getenv("GROQ_KEY2")],  # account 1
    [os.getenv("GROQ_KEY3"), os.getenv("GROQ_KEY4")],  # account 2
]

class ModelPool:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.dead_keys = set()
        self.current_pool = 0
        self.current_key_index = 0
        self.model = self._init_model()

    def _init_model(self):
        while self.current_pool < len(POOLS):
            keys = POOLS[self.current_pool]
            while self.current_key_index < len(keys):
                key = keys[self.current_key_index]
                if key not in self.dead_keys:
                    return ChatGroq(model=self.model_name, api_key=key)
                self.current_key_index += 1
            self.current_pool += 1
            self.current_key_index = 0
        raise RuntimeError("Groq quota exhausted across all accounts")

    def mark_dead(self):
        self.dead_keys.add(self.model.api_key)
        self.current_key_index += 1
        self.model = self._init_model()
