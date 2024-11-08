from math import *
from ta import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import yfinance as yf
#all imports
price = yf.ticker('AAPL').history(period='1y')['Close']
