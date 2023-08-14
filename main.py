import math
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
from numpy import median
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from statsmodels.stats.weightstats import ztest
import csv
import cv2
print(cv2.__version__)
#############################
#GETTING GRAIN SIZES
with open('Average_Grain_Size_Distributions.csv', newline='') as f:
  reader = csv.reader(f)
  row1 = next(reader)  # gets the first line
medians_sample_2 = row1[9:]
for i in range(len(medians_sample_2)):
    medians_sample_2[i] = float(medians_sample_2[i])

#############################
df = pd.read_csv('results.csv')
diameters = df.loc[:,"EquivDiameter"]
diameters = [float(i) * 0.81 for i in diameters]
imgNum = df.loc[:,"file"]

imgNumPlt = []
medians_sample_1 = []
samples_84th_percent = []
samples_16th_percent = []
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
        median_num = statistics.median(lister)
        medians_sample_1.append(median_num)
        #ADD 84th and 16th percentile values
        samples_84th_percent.append(np.quantile(lister, 0.84))
        samples_16th_percent.append(np.quantile(lister, 0.16))
        ####################################
        count2 += 1
        lister.clear()
        lister.append(diameters[i])

lister.sort() #doesnt catch the last number
medians_sample_1.append(statistics.median(lister))
samples_84th_percent.append(np.quantile(lister, 0.84))
samples_16th_percent.append(np.quantile(lister, 0.16))

imgNumPlt = np.arange(len(medians_sample_1))
############################
#PLOTTING CDF OF DISTRIBUTION
CDF_1 = []
CDF_2 = []
medians_sample_1.sort()
medians_sample_2.sort()

size_1 = len(medians_sample_1)
size_2 = len(medians_sample_2)
np_size_1 = np.array(medians_sample_1)
np_size_2 = np.array(medians_sample_2)

for i in np_size_1:
    temp1 = np_size_1[np_size_1 <= i]
    value = temp1.size / size_1
    # pushing the value in the y_values
    if value not in CDF_1:
        CDF_1.append(value)

for i in np_size_2:
    temp2 = np_size_2[np_size_2 <= i]
    value = temp2.size / size_2
    # pushing the value in the y_values
    if value not in CDF_2:
        CDF_2.append(value)

x_axis_1 = np.arange(len(CDF_1))
x_axis_2 = np.arange(len(CDF_2))

plt.xlabel('Diameter (um)')
plt.ylabel('Probability in Decimal')
plt.title('CDF Plot')

plt.plot(x_axis_1, CDF_1, label = "Sample 1")
plt.plot(x_axis_2, CDF_2, label = "Sample 2")
plt.legend()
plt.xscale("log")
plt.show()

############################
#PLOTTING PDF OF DISTRIBUTION
plt.close()
#TODO: CHANGE X AXIS TO DIAMETER

# fig, ax = plt.subplots(figsize =(10, 7))
# ax.hist(x_axis, bins = [0, 4, 8, 12])
bins = [0, 4, 8]
plt.hist(medians_sample_1, bins= 10, label='Sample 1')
plt.hist(medians_sample_2, bins = 10, label='Sample 2')
plt.legend(loc='upper right')

print(medians_sample_1)
print(medians_sample_2)

#plt.plot(x_axis, my_list)
plt.xlabel('Diameter (um)')
plt.ylabel('Number of Samples')
plt.title('PDF Plot')
plt.xscale("log")
plt.show()

#############################
#PLOTTING T TEST
t_stat, p_value = ttest_ind(medians_sample_1, medians_sample_2)
print("T-statistic value: ", t_stat)
print("P-Value: ", p_value)

#Kolmogorov-Smirnov test
print(ks_2samp(medians_sample_1, medians_sample_2))

#Z Test
test_statistic, p_value = ztest(medians_sample_1, medians_sample_2, value=0)
print("Z Test Statistic:", test_statistic, "| Z P_Value:", p_value)


#FINDING THE # OF PARTICLES PER IMAGE
df = pd.read_csv('results.csv')
imgNum = list(df.loc[:,"file"])
total_num_particles = {}
list_of_elements = []
for i in imgNum:
    if i not in list_of_elements:
        list_of_elements.append(i)
for i in list_of_elements:
    total_num_particles[i] = imgNum.count(i)

figure, axis = plt.subplots(2, 1)

imgNumPlt = list(imgNumPlt)
for i in range(len(imgNumPlt)):
    imgNumPlt[i] *= 60 #60 FPS, converting to time

axis[0].plot(imgNumPlt, total_num_particles.values())
axis[0].set_title('Total Measured Particles and Median Size Particle per Image')
#axis[0].set_xlabel('')
axis[0].set_ylabel('Number of Particles')
#axis[0].set_xscale("log")

axis[1].plot(imgNumPlt, medians_sample_1)
#axis[1].set_title('Median')
axis[1].set_xlabel('Time (seconds)')
axis[1].set_ylabel('Diameter (um)')
#axis[1].set_xscale("log")

plt.show()
#PLOTTING PSD VS GSD


######################
#PLOTTING THE 16TH, 50TH, AND 84TH PERCENTINE FOR DIAMETERS
plt.plot(imgNumPlt, medians_sample_1, label = "50th Percentile")
plt.plot(imgNumPlt, samples_16th_percent, label = "16th Percentile")
plt.plot(imgNumPlt, samples_84th_percent, label = "84th Percentile")

plt.title('Image Diameter Percentiles')
plt.xlabel('Time (Seconds)')
plt.ylabel('Diameter (um)')
plt.legend()
plt.show()

