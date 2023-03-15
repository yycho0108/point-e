#!/usr/bin/env python3

import os
from typing import Optional, Iterable
from textwrap import dedent
import openai

GPT_VERSION: str = 'gpt-3.5-turbo'
API_KEY= os.getenv('OPENAI_API_KEY')
openai.api_key = API_KEY


def LM(prompt: str, max_tokens: int = 128,
        temperature: float = 0,
        stop: Optional[Iterable[str]] = None):
    print(F'prompt = [{prompt}]')
    response = openai.Completion.create(
        engine=GPT_VERSION,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop)
    return response['choices'][0]['text'].strip()


prompt = dedent("""\
        # Q: Household object that can be improvised to reach a fallen toy in a gap.
        # A: Ruler.

        # Q: Household object that can be improvised to open a wine bottle.
        A: """)
LM(prompt, stop=['#'])
