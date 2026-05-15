import pandas as pd
print("Saving test csv...")
df = pd.DataFrame({'test': [1, 2, 3]})
df.to_csv('test_save_direct.csv', index=False)
print("Done!")
