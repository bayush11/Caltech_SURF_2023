#This file takes in a pixel value and converts it to a real life measurement 
#depending on the conversion reate that is used. This number can be changed in 
#line 12 to whatver your real life change is.

import pandas as pd

df = pd.read_csv('results_CSV_2.csv')
diameters = df.loc[:,"EquivDiameter"]
new_diameters = []

for i in diameters: #this loop is responsible for converting the pixel to the real life measurment.
    new_diameters.append(i * 3.14151617 - 2) #random conversion for now

df['pixel_to_real'] = new_diameters
df.to_csv('results_CSV_2.csv') #exporting the results to a csv file.
