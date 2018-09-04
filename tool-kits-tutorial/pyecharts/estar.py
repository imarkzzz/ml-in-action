from pyecharts import Bar
import pandas as pd
import numpy as np


def plot_bar(dump=False):
  attr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Aug", "Sep", "Oct", "Nov", "Dec"]
  precipitation = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
  evaporation = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
  bar = Bar("Bar chart", "precipitation and evaporation one year")
  bar.add("precipitation", attr, precipitation, mark_line=["average"], mark_point=["max", "min"])
  bar.add("evaporation", attr, evaporation, make_line=["average"], make_point=["max", "min"])
  bar.render()
  if dump:
    bar.render(path="render.png")


def plot_bar_with_rnd_gen_data(dump=False):
  title = 'bar chart'
  parts = 6
  index = pd.date_range("9/4/2018", periods=parts, freq="M")
  profit_df = pd.DataFrame(np.random.randn(parts), index=index)
  loss_df = pd.DataFrame(np.random.randn(parts), index=index)

  profit_dt_val = [i[0] for i in profit_df.values]
  loss_dt_val = [i[0] for i in loss_df.values]
  _index = [i for i in profit_df.index.format()]
  
  bar = Bar(title, "Profit and loss situation")
  bar.add('profit', _index, profit_dt_val)
  bar.add('loss', _index, loss_dt_val)
  bar.render()
  if dump:
    bar.render(path="profit-loss.png")

def main():
  plot_bar()

if __name__ == '__main__':
  main()