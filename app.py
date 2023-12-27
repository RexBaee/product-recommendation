import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

#load dataset
df = pd.read_csv('Bakery sales.csv')

df["date"] = pd.to_datetime(df["date"] + " " + df["time"])
df.rename(columns={"date":"date_time"},inplace=True)

df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M:%S")

df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.weekday

df["month"].replace([i for i in range(1,13)],["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],inplace=True)
df["day"].replace([i for i in range(7)],["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],inplace=True)

st.title("Recommendation Bakery Products")

st.markdown("This application is a Streamlit dashboard that can be used ")

def get_data(month = '', day = ''):
  data = df.copy()
  if month and day:
      filtered = data[(data["month"] == month) & (data["day"] == day)]
      return filtered if not filtered.empty else "No Result"
  elif month:
      filtered = data[data["month"] == month]
      return filtered if not filtered.empty else "No Result"
  elif day:
      filtered = data[data["day"] == day]
      return filtered if not filtered.empty else "No Result"
  else:
      return "No Result"

def user_input_features():
  item = st.selectbox("Item",df["article"].unique())
  month = st.select_slider("Month",["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], value="Jan")
  day = st.select_slider("Day",["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], value="Mon")
  return month, day, item

month, day, item = user_input_features()

data = get_data(month, day)

def encode_units(x):
  if x <= 0:
    return 0
  elif x >= 1:
    return 1

if (type(data) != type ("No Result")):
  item_count = data.groupby(["ticket_number","article"])["Quantity"].sum().reset_index(name = "Count")
  item_count_pivot = item_count.pivot_table(index="ticket_number",columns="article",values="Count",aggfunc="sum").fillna(0)
  item_count_pivot = item_count_pivot.applymap(encode_units)

  support = 0.01
  frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

  matrix = 'lift'
  min_threshold = 1

  rules = association_rules(frequent_items, metric=matrix, min_threshold=min_threshold)[["antecedents","consequents","support","confidence","lift"]]
  rules.sort_values('confidence',ascending=False,inplace=True)

def parse_list(x):
  x = list(x)
  if len(x) == 1:
    return x[0]
  elif len(x) > 1:
    return ", ".join(x)

def return_item_df(item_antecedents):
  data_filtered = rules[(rules['antecedents'].apply(lambda x: item_antecedents in x))]
  if not data_filtered.empty:
    antecedent = parse_list(data_filtered['antecedents'].values[0])
    consequent = parse_list(data_filtered['consequents'].values[0])
    return [antecedent, consequent]
  else:
    return ["No Result", "Yo Ndak Tahu"]

if (type(data) != type ("No Result")):
  if (return_item_df(item)[1] != "Yo Ndak Tahu"):
    st.markdown("Hasil rekomendasi : ")
    st.success(f"Jika Konsumen membeli {item}, maka konsumen juga akan membeli {return_item_df(item)[1]} secara bersamaan")
  else:
    st.markdown("Konsumen hanya membeli satu item")
else:
  st.markdown("Konsumen hanya membeli satu item")

print(df.head())
