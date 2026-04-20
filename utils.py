# transform GPT-generated
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import subprocess
import ast

def clean_circ(tokens):
    """
    Clean GPT-generated QAOA circuit tokens into values that ADAPT.jl can parse.
    Handles special tokens (new_layer_p, eos) and cleans operator strings (op_123 -> 123).
    """
    cleaned = []
    # Special tokens to keep
    SPEC_KEEP = ['new_layer_p']
    # Special tokens to discard
    SPEC_DROP = ['bos', 'eos', 'pad', 'end_of_formula', 'end_of_graph']
    
    for t in tokens:
        t_str = str(t)
        if t_str in SPEC_DROP:
            continue
        if t_str in SPEC_KEEP:
            cleaned.append(t_str)
            continue
            
        if t_str.startswith('op_'):
            try:
                cleaned.append(int(t_str.split('_')[1]))
            except Exception:
                cleaned.append(t_str)
        else:
            try:
                # Try to parse as float (angles)
                cleaned.append(float(t_str))
            except Exception:
                cleaned.append(t_str)
    return cleaned
