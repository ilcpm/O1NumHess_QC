from .O1NumHess_QC import *
from pathlib import Path

# TODO 首次运行时检测是否存在配置文件，若不存在，生成模版
config_folder = Path("~/.O1NumHess_QC").expanduser().absolute()
