'''
This function adds irregularit to the strings.
Usage: add_irregularities('Hello', seed=42)
'''

import random, re

def add_irregularities(text, seed=None):
    rng = random.Random(seed) if seed is not None else random
    text = text.lower() if rng.random() < 0.5 else text.upper()  # random casing
    if rng.random() < 0.5:
        text = re.sub(r'', rng.choice(['_','-',' ']), text, count=rng.randint(1,3))
    else:
        text += rng.choice(['12','65','23','156','189'])
        text += rng.choice(['@','<>','()','# ','  '])
    return text
