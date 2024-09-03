from importlib_resources import files
from pandas import read_csv

schedule_file = files("BlendPATH.util").joinpath("pipe_dimensions_metric.csv")

SCHEDULES = read_csv(schedule_file, index_col=None, header=0)
