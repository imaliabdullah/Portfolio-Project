import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
import bisect
import collections

# store the url in variable
url = "https://oldschool.runescape.wiki/w/Zulrah"

# access the contents of the webpage
html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

# extract HTML from response
page_content = html.content

# parse html
soup = BeautifulSoup(page_content, 'lxml')

# find all tables in the html
my_tables = soup.find_all('table', attrs = {'class': 'wikitable sortable filterable item-drops autosort=4,a'})

# initialize empty dictionary
loot_tables = {}
table_index = 0

for table in my_tables:
    # parse out table rows
    table_rows = table.find_all('tr')

    # parse out the table header
    table_header = table.find_all('th')

    # parse columns from table header
    cols = []
    for item in table_header:
        if len(item.text) > 0:
            cols.append(item.text)

    # parse data from table
    data = []
    for row in table_rows:
        table_data = row.find_all('td')

        row_data = [item.text for item in table_data if item.text != '']
        if len(row_data) > 0:
            data.append(row_data)
    
    # create data frame
    output_frame = pd.DataFrame(data, columns=cols)

    loot_tables.setdefault(table_index, output_frame)
    table_index += 1

loot_tables = pd.concat(loot_tables.values())
loot_tables.set_index("Item", inplace=True)
