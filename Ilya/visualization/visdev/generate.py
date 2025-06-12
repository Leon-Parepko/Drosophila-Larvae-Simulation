from activity_generator import *

w, a = generate_activity_df(some_activity_system(), 1000)

w.to_csv('Datasets/Generated/xwx_activity/weights.csv')
a.to_csv('Datasets/Generated/xwx_activity/activity.csv')