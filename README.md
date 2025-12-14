### Box Office Analysis for Strategic Movie Production
This notebook contains a full end-to-end exploratory data analysis (EDA) combining IMDB and Box Office Mojo data to inform a new movie studio's production strategy.
It includes data loading, cleaning, merging (exact + fuzzy matching), EDA visualizations, and three concrete business recommendations.

# Importing Data and Settings


```python
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from difflib import get_close_matches
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['font.size'] = 12
```

## 2) Paths and helper functions


```python
DATA_DIR = '/mnt/data'
IMDB_DB = os.path.join(DATA_DIR, 'im.db')
BOM_FILE = os.path.join(DATA_DIR, 'bom.movie_gross.csv.gz')
OUTPUT_DIR = os.path.join(DATA_DIR, 'eda_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_money_column(s):
    return s.astype(str).str.replace(r'[$,]', '', regex=True).replace('', np.nan)

def save_fig(fig, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, bbox_inches='tight')
```

## 3) Load data


```python
print('Loading IMDB database...')
conn = sqlite3.connect(IMDB_DB)
movie_basics = pd.read_sql('SELECT * FROM movie_basics;', conn)
movie_ratings = pd.read_sql('SELECT * FROM movie_ratings;', conn)
conn.close()
print('Loading BOM...')
bom = pd.read_csv(BOM_FILE, compression='gzip', low_memory=False)
print('Done. Rows:', len(movie_basics), 'IMDB basics; ', len(bom), 'BOM rows')

```

4. chech


```python

```
