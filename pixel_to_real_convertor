import pandas as pd

df = pd.read_csv('results_CSV_2.csv')
diameters = df.loc[:,"EquivDiameter"]
new_diameters = []

for i in diameters:
    new_diameters.append(i * 3.14151617 - 2) #random conversion for now

df['pixel_to_real'] = new_diameters
df.to_csv('results_CSV_2.csv')
