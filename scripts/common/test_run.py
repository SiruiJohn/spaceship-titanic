import pandas as pd
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
df.to_csv('test_save.csv', index=False)
print("Saved test_save.csv")
