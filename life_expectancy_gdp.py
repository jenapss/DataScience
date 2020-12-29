import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


csv = pd.read_csv('all_data.csv')

csv.hist(bins = 120,)