import pandas as pd
# ['comment_count', 'content', 'created_date', 'like_count',
# 'sentiment', 'share_count', 'source_type']

data =pd.read_csv("newdata.csv")
print(list(data))
writer = pd.ExcelWriter('newdata.xlsx')

data.head(100).to_excel(writer,'Sheet1')
writer.save()
