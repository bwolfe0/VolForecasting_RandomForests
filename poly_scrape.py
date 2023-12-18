import pandas as pd
import polygon as pg
import matplotlib.pyplot as plt
import datetime as dt
from secret import polygon_key

C = pg.OptionsClient(polygon_key)
RC = pg.reference_apis.reference_api.ReferenceClient(polygon_key)

u_tick = 'SPY'
strike = '00472000'
type = 'C'
date = '231218'

tick = 'O:SPY'+date+type+strike

print(