# Group 4 Phase 2 Project
### Student name: Monica Wairimu, Jeffrey Gathigi, Jared Mongeri, John Awallah
### Student Pace: Part-time
### Instructor Name: Brianc Chacha and Lians Wanjiku

# Box Office Analysis for Strategic Movie Production
This notebook contains a full end-to-end exploratory data analysis (EDA) combining IMDB and Box Office Mojo data to inform a new movie studio's production strategy.
It includes data loading, cleaning, merging, EDA visualizations, and three concrete business recommendations.

### Objectives of the project
1. **Identify which movie genres generate the highest worldwide revenue in order to guide the studio’s decisions on which types of films to prioritize.**  
2. **Analyze the relationship between IMDB ratings and worldwide box office revenue to understand how audience approval impacts financial performance**  
3. **Examine how movie runtime relates to worldwide revenue to determine the optimal runtime range for commercially successful films**

# 1. Importing Data and Settings


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

# 2. Unzipping the imdb dataset


```python
import zipfile
with zipfile.ZipFile('./zippedData/im.db.zip', 'r') as z:
    z.extractall('.')
```


```python
print(os.path.abspath('im.db'))
print(os.path.getsize('im.db'), bytes)
```

    c:\Users\USER\Documents\Moringa\Project_phase_3\group_4_project_phase_2\im.db
    169443328 <class 'bytes'>
    

# 3. Loading Data
### Data will be loaded from the following;
1. IMDB SQLite database (`im.db`)
2. Box Office Mojo dataset (`bom.movie_gross.csv.gz`)


```python
conn = sqlite3.connect('im.db')

movie_basics = pd.read_sql_query("SELECT * FROM movie_basics", conn)
movie_ratings = pd.read_sql_query("SELECT * FROM movie_ratings", conn)

bom = pd.read_csv('./zippedData/bom.movie_gross.csv.gz') #box office mojo dataset

movie_basics.head(), movie_ratings.head(), bom.head()
```




    (    movie_id                    primary_title              original_title  \
     0  tt0063540                        Sunghursh                   Sunghursh   
     1  tt0066787  One Day Before the Rainy Season             Ashad Ka Ek Din   
     2  tt0069049       The Other Side of the Wind  The Other Side of the Wind   
     3  tt0069204                  Sabse Bada Sukh             Sabse Bada Sukh   
     4  tt0100275         The Wandering Soap Opera       La Telenovela Errante   
     
        start_year  runtime_minutes                genres  
     0        2013            175.0    Action,Crime,Drama  
     1        2019            114.0       Biography,Drama  
     2        2018            122.0                 Drama  
     3        2018              NaN          Comedy,Drama  
     4        2017             80.0  Comedy,Drama,Fantasy  ,
          movie_id  averagerating  numvotes
     0  tt10356526            8.3        31
     1  tt10384606            8.9       559
     2   tt1042974            6.4        20
     3   tt1043726            4.2     50352
     4   tt1060240            6.5        21,
                                              title studio  domestic_gross  \
     0                                  Toy Story 3     BV     415000000.0   
     1                   Alice in Wonderland (2010)     BV     334200000.0   
     2  Harry Potter and the Deathly Hallows Part 1     WB     296000000.0   
     3                                    Inception     WB     292600000.0   
     4                          Shrek Forever After   P/DW     238700000.0   
     
       foreign_gross  year  
     0     652000000  2010  
     1     691300000  2010  
     2     664300000  2010  
     3     535700000  2010  
     4     513900000  2010  )



# 4. Exploring the Schema of the Data


```python
#Exploring the Imdb Dataset
# List tables
conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
```




    [('movie_basics',),
     ('directors',),
     ('known_for',),
     ('movie_akas',),
     ('movie_ratings',),
     ('persons',),
     ('principals',),
     ('writers',)]




```python
# Schema for movie_basics from imdb dataset
conn.execute("PRAGMA table_info(movie_basics);").fetchall()
```




    [(0, 'movie_id', 'TEXT', 0, None, 0),
     (1, 'primary_title', 'TEXT', 0, None, 0),
     (2, 'original_title', 'TEXT', 0, None, 0),
     (3, 'start_year', 'INTEGER', 0, None, 0),
     (4, 'runtime_minutes', 'REAL', 0, None, 0),
     (5, 'genres', 'TEXT', 0, None, 0)]




```python
# Schema for movie_ratings
conn.execute("PRAGMA table_info(movie_ratings);").fetchall()
```




    [(0, 'movie_id', 'TEXT', 0, None, 0),
     (1, 'averagerating', 'REAL', 0, None, 0),
     (2, 'numvotes', 'INTEGER', 0, None, 0)]




```python
#Schema for the bom Data
bom.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3387 entries, 0 to 3386
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           3387 non-null   object 
     1   studio          3382 non-null   object 
     2   domestic_gross  3359 non-null   float64
     3   foreign_gross   2037 non-null   object 
     4   year            3387 non-null   int64  
    dtypes: float64(1), int64(1), object(3)
    memory usage: 132.4+ KB
    

# 5. Data Cleaning
This is to ensure there are clean titles to ensure an easy merge
Further, since the bom data is about movies and not tv shows, we have filtered the imdb data to remove items with tv show like characteristic 


```python
# Merge IMDB tables first
imdb = movie_basics.merge(movie_ratings, on='movie_id', how='left')
# Apply movie-like filters
imdb = imdb[
    (imdb['runtime_minutes'] > 40) &
    (imdb['numvotes'] > 1000) &
    (imdb['genres'].notna())
]

len(imdb)
# Clean IMDB titles
imdb['clean_title'] = (
    imdb['primary_title']
    .str.lower()
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace(r'[^a-z0-9 ]', '', regex=True)
    .str.strip()
)
imdb['clean_title'].head()
```




    2     the other side of the wind
    20                     foodfight
    32                    dark blood
    38                         wazir
    39               the wicker tree
    Name: clean_title, dtype: object




```python

```
