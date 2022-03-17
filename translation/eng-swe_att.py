import numpy as np

import typing
from typing import Any, Tuple

import tensorflow as tf

import tensorflow_text as tf_text
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lines= pd.read_table('dataset.txt',  names =['inp', 'targ', 'comments'])

BUFFER_SIZE = len(lines.inp)
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((lines.inp, lines.targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

for example_input_batch, example_target_batch in dataset.take(1):
  print(example_input_batch[:10])
  print()
  print(example_target_batch[:5])
  break
