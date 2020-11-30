import pandas as pd

import emoji
import re
import numpy as np
import pandas as pd
from autocorrect import Speller

spell = Speller(lang='en')

def preprocess(inp):
    x = emoji.demojize(inp)
    x= re.sub(r"Image:.*(?:.jpg|.svg|.png|.jpeg|.gif|.tif|.ext)",' ',x,flags=re.IGNORECASE)
    x= re.sub(r"File:.*(?:.jpg|.svg|.png|.jpeg|.gif|.tif|.ext)",' ',x,flags=re.IGNORECASE)
    x= re.sub(r'[^a-zA-Z\s]',' ',x)
    x= re.sub(r'[\n]',' ',x) 
    #x = spell(inp)
    return str(x)   

train = pd.read_csv('input/train.csv')
train['comment_text'] = train['comment_text'].fillna('')
print("fill done")

train['comment_text'] = train['comment_text'].apply(lambda x: preprocess(x))
train.to_csv('cleaned/train.csv', index=False)
