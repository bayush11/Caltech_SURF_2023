#This code file will help to discuss the creation of the GSD table, which willl help calculating variables like the individual and combined GSD. 
#After this, the file discusses how the GSD and PSD and calculated in PDF/CDF format.


import csv
import math
import statistics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp

grain_types = {} #This part of the code is used to get the grain size distribution into the file.
with open('Average_Grain_Size_Distributions.csv', newline='') as f:
  reader = csv.reader(f)
  row1 = next(reader)
  grain_size = [float(i) for i in row1[1:]]

  for i in range(3):
      row1 = next(reader)
      grain_types[row1[0]] = [float(i) for i in row1[1:]]

grain_types_names = list(grain_types.keys())

table = {} #These sections are going to store the information where each table section is.
table['sediment'] = []
table['class number'] = []
table['D Lower'] = []
table['D Upper'] = []
table['D Center'] = []
table['Volume Fraction'] = []
index_number = 1
for i in range(3):
    class_number = 1
    curr_grain_type = grain_types_names[i] #this line and the one below get the grain type and values that will be used to get table values.
    curr_grain_vals = grain_types[curr_grain_type]
    for j in range(len(curr_grain_vals) - 1): #n-1 iterations as we are taking the class
        table['sediment'].append(curr_grain_type) #these are the calculations done below to get the specific table values.
        table['class number'].append(class_number)
        table['D Lower'].append(grain_size[j])
        table['D Upper'].append(grain_size[j + 1])
        center_num = math.exp((math.log(table['D Lower'][index_number - 1]) + math.log(table['D Upper'][index_number - 1])) / 2)
        table['D Center'].append(center_num)
        table['Volume Fraction'].append(curr_grain_vals[j + 1] - curr_grain_vals[j])
        class_number += 1
        index_number += 1

###############
proportion_s = 0.6 #These are the invidual GSD proprtions I made up for each grain type.
proportion_k = 0.2
proportion_m = 0.2
list_of_proportions = [proportion_s, proportion_k, proportion_m]
dict_of_proportions = {}
for i in range(len(grain_types_names)): #this sets the list of proportions to each of the grain types.
    dict_of_proportions[grain_types_names[i]] = list_of_proportions[i]
#GOAL: TRY TO GET THE FACTIONAL VOL FOR EACH CLASS #
vol_fractions_grains = {}
sediments = table['sediment']
vol_frac = table['Volume Fraction']

first_sediment_name = sediments[0]
num_of_sediments = 0
for i in range(len(sediments)): #gets the number of sediments for one type; this gives insight to the other types
    if sediments[i] == first_sediment_name:
        num_of_sediments += 1

table_counter = 0
for i in range(3): #this is to get lists of the different vol_fracs
    name = grain_types_names[i]
    vol_fractions_grains[name] = []
    for j in range(num_of_sediments):
        vol_fractions_grains[name].append(vol_frac[table_counter])
        table_counter += 1
##################
#Finding the combined y_axis value
x_axis = np.arange(num_of_sediments)
y_axis_combined_GSD = []
for i in range(num_of_sediments): #the code below will get the combined GSD values and store that into the list for this.  
    val_of_y = (proportion_s * vol_fractions_grains[grain_types_names[0]][i]) + proportion_k * vol_fractions_grains[grain_types_names[1]][i] + proportion_m * vol_fractions_grains[grain_types_names[2]][i]
    num = val_of_y / (math.log(table['D Upper'][i]) - math.log(table['D Lower'][i]))
    y_axis_combined_GSD.append(num)
total_GSD_value = sum(y_axis_combined_GSD)
for i in range(num_of_sediments): #the normalizing of the combined GSD values
    normalized_GSD = y_axis_combined_GSD[i] / total_GSD_value
    y_axis_combined_GSD[i] = normalized_GSD

#GRAPH COMBINED GSD
plt.xlabel('Grain Size (um)') #plotting the results of the combined GSD with a log scale for the x axis.
plt.ylabel('PDF')
plt.title('Different Grain Size Distribution Plot')
plt.xscale("log")
plt.plot(x_axis, y_axis_combined_GSD, label = "Combined GSD")
###################
keys = list(table.keys())
list_of_GSD_dict = []
for i in range(len(sediments)): #new format for the table values as this will be exported to a csv.
    new_dict = {}
    new_dict[keys[0]] = table[keys[0]][i]
    new_dict[keys[1]] = table[keys[1]][i]
    new_dict[keys[2]] = table[keys[2]][i]
    new_dict[keys[3]] = table[keys[3]][i]
    new_dict[keys[4]] = table[keys[4]][i]
    new_dict[keys[5]] = table[keys[5]][i]
    list_of_GSD_dict.append(new_dict)


with open('table_results.csv', 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = keys) #exporting to a seperate csv.
    writer.writeheader()
    writer.writerows(list_of_GSD_dict)
###################################################
#CREATING INDIVIDUAL GRAPHS OF THE SEDIMENTS
individual_sediment_pdf = {}
for name in list(vol_fractions_grains.keys()):
   individual_sediment_pdf[name] = []
   proportion = dict_of_proportions[name]
   for i in range(len(vol_fractions_grains[name])): #This is done to find the individual grain size distribution values as before it was the combined GSD.
       f = vol_fractions_grains[name][i]
       num = f / (math.log(table['D Upper'][i]) - math.log(table['D Lower'][i]))
       individual_sediment_pdf[name].append(num)
   sum_of_vals = sum(individual_sediment_pdf[name])
   for i in range(len(individual_sediment_pdf[name])):
       individual_sediment_pdf[name][i] = individual_sediment_pdf[name][i] / sum_of_vals

x_axis = np.arange(num_of_sediments) #plotting these values.
for name_sediment, y_axis in individual_sediment_pdf.items():
    plt.plot(x_axis, y_axis, label = "GSD of " + str(name_sediment))

plt.legend()
plt.show()

###################
#PLOTTING GSD AND PSD
plt.close()
df = pd.read_csv('results.csv')
diameters = list(df.loc[:,"EquivDiameter"])
diameters = [float(i) * 0.81 for i in diameters]
CDF_for_GSD = []
for i in range(len(y_axis_combined_GSD)): #finding the CDF of the GSD.
    CDF_for_GSD.append(sum(y_axis_combined_GSD[0:i]))
#CDF OF PSD
CDF_for_PSD = []
diameters.sort()
size_data = len(diameters)
np_diameters = np.array(diameters)
for i in diameters:
    temp = np_diameters[np_diameters <= i]
    # fraction of that value with respect to the size of the x_values
    value = temp.size / size_data
    # pushing the value in the y_values
    if value not in CDF_for_PSD: #finding the CDF for the PSD
        CDF_for_PSD.append(value) 
##################

x_axis_1 = list(np.arange(len(CDF_for_PSD))) #plotting the CDF values for the GSD and the PSD.
x_axis_2 = list(np.arange(len(CDF_for_GSD)))
plt.plot(x_axis_1, CDF_for_PSD, label = "PSD")
plt.plot(x_axis_2, CDF_for_GSD, label = "GSD")
plt.xscale("log")
plt.title('PSD vs GSD')
plt.xlabel('Diameter (um)')
plt.ylabel('CDF')
plt.legend()
plt.show()

################################
#Plotting the GSD vs PSD in PDF format
#THIS IS ALL FINDING PSD MEDIAN
df = pd.read_csv('results.csv')
diameters = df.loc[:,"EquivDiameter"]
diameters = [float(i) * 0.81 for i in diameters]
imgNum = df.loc[:,"file"]

imgNumPlt = []
medians_sample_1 = []
lister = []
count2 = 0

for i in range(len(imgNum)):
    if imgNum[i] not in imgNumPlt:
        imgNumPlt.append(imgNum[i])

for i in range(len(imgNum)):
    if imgNum[i] == imgNumPlt[count2]:
        lister.append(diameters[i])
    else:
        lister.sort()
        median_num = statistics.median(lister) #finding the PDF of the PSD.
        medians_sample_1.append(median_num)
        count2 += 1
        lister.clear()
        lister.append(diameters[i])

lister.sort() #doesnt catch the last number
medians_sample_1.append(statistics.median(lister))

###############################
(n, bins, patches) = plt.hist(medians_sample_1, density = True, bins= 15, label='PSD') #used to get info about PSD

#this is to find the cutoff version of the GSD data
min_diameter = min(medians_sample_1)
y_axis_GSD = []
x_axis_2 = []
curr_diameters = table['D Center'][:100]
for i in range(len(curr_diameters)):
    if curr_diameters[i] >= min_diameter:
        y_axis_GSD.append(y_axis_combined_GSD[i])
        x_axis_2.append(curr_diameters[i]) #cutting off the values of the GSD less than the minimum PSD value as those values wont serve any help in seeing flocculation trends.

sum = sum(y_axis_GSD)
for i in range(len(y_axis_GSD)):
    y_axis_GSD[i] = y_axis_GSD[i] / sum

plt.plot(x_axis_2, y_axis_GSD, label = "GSD") #plotting the results.

plt.legend(loc='upper right')
plt.xscale("log")
plt.title('PSD vs GSD')
plt.xlabel('Diameter (um)')
plt.ylabel('PDF')
plt.legend()
plt.show()

print(ks_2samp(medians_sample_1, y_axis_GSD))
