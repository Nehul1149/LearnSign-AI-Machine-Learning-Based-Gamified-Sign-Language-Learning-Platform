import pandas as pd

df = pd.read_csv("hand_gestures.csv")
print(df['label'].value_counts())
