import pandas as pd
# ['comment_count', 'content', 'created_date', 'like_count',
# 'sentiment', 'share_count', 'source_type']

data =pd.read_json("newdata.json")
data.to_csv("newdata.csv")
