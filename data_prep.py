import pandas as pd
import numpy as np

import zipfile
import glob
import json
import os

season_stats1415 = json.load(open('data/season14-15/season_stats.json','r'))
print("{}".format(json.dumps(season_stats1415,indent=2))[:3000])
