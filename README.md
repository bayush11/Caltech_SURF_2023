# SURF CALTECH 2023 README

# Background

Below I will be discussing the particle tracking project that was being done over a 10-week course at Caltech summer 2023. The main goal of this project is to track particles to figure out their velocity and size. I will give insight onto any results that I have gotten as well as provide code to help future readers of this document. The data of the grains and particles that will be discussed are available due to the work of Kim and Kyd who conducted particle experiments. Some important information to know is that GSD stands for grain size distribution and represents the sizes of the grains at the beginning of the experiments, while PSD is the particle size distribution and represents the particle sizes throughout the experiment.

# Imaging Data 

The imaging data was processed through the Keyvani and Strom (2013) workflow. This is the paper link: https://doi.org/10.1016/j.cageo.2012.08.018.

Below are the different phases that the images have gone through in the Keyvani and Strom (2013) workflow. When processing a time-series, one should get folders of these three image types plus a binarized labeled image.

# Greyscale Image
![frame_0001](https://github.com/bayush11/Caltech_SURF_2023/assets/70395352/2f34e207-d9e8-4aa6-b74f-c4baba509813)
# Gradient Image
![frame_0001](https://github.com/bayush11/Caltech_SURF_2023/assets/70395352/79ee9728-90b1-44b8-afe6-64720713e9d9)
# Binarized Image
![frame_0001](https://github.com/bayush11/Caltech_SURF_2023/assets/70395352/84fc46c2-b1da-4887-bb5a-53f3eca1f9a0)

# Frame vs Diameter Code

Before going into the code, this code snippet below shows how to convert from the frame number to time. The FPS is 60, so this is just a multiplication factor that is needed.

```bash
# Frame to Seconds
for i in range(len(imgNumPlt)):
    imgNumPlt[i] *= 60 
```
Below is the code that plots the frame by the diameter of the flocs. When using the Keyvani and Strom (2013) workflow, one will receive a CSV file that has the diameters of every particle tracked in the binarized images. This code will take that CSV and find the median diameter per frame. **THIS IS IN THE MAIN.PY FILE.**

This code snippet helps to get the number of frames there are in the csv file.
```bash
for i in range(len(imgNum)):
    if imgNum[i] not in imgNumPlt:
        imgNumPlt.append(imgNum[i])
```
This is where using the unique numbers, one finds the median diameter per frame.
```bash
for i in range(len(imgNum)):
    if imgNum[i] == imgNumPlt[count2]:
        lister.append(diameters[i])
    else:
        lister.sort()
        median_num = statistics.median(lister)
        medians_sample_1.append(median_num)
        count2 += 1
        lister.clear()
        lister.append(diameters[i])

lister.sort() #doesnt catch the last number
medians_sample_1.append(statistics.median(lister))
```
This is now plotting the results.
```bash
axis[1].plot(imgNumPlt, medians_sample_1)
```
Results
![median_img](https://github.com/bayush11/Caltech_SURF_2023/assets/70395352/5f6d0a59-2a40-47d4-906e-307c1f5e864c)

The code below describes how to get the total number of particles per frame. This trend can potentially suggest flocculation as if the number of particles is decreasing, then that shows particles are combining, therefore reducing the total number of them overall and showing flocculation.

```bash
df = pd.read_csv('results.csv')
imgNum = list(df.loc[:,"file"])
total_num_particles = {}
list_of_elements = []
for i in imgNum:
    if i not in list_of_elements:
        list_of_elements.append(i)
for i in list_of_elements:
    total_num_particles[i] = imgNum.count(i)
```
    

# Table Creation

The table that I will be discussing is a file that I created to list the many important features regarding grain size distribution information. This table will be used throughout the rest of the code as the information is relevant.

**FURTHER INFORMATION CAN BE FOUND IN THE table_creator.py FILE**

```bash
#This information is later exported into a separate CSV file
table = {}
table['sediment'] = []
table['class number'] = []
table['D Lower'] = []
table['D Upper'] = []
table['D Center'] = []
table['Volume Fraction'] = [] #these all init the table
index_number = 1
for i in range(3): #3 because there are three starting grain types
    class_number = 1
    curr_grain_type = grain_types_names[i]
    curr_grain_vals = grain_types[curr_grain_type] #getting the grain name and values associated
    for j in range(len(curr_grain_vals) - 1): #n-1 iterations as we are taking the class
        table['sediment'].append(curr_grain_type) #this now the math done to get the wanted values
        table['class number'].append(class_number)
        table['D Lower'].append(grain_size[j])
        table['D Upper'].append(grain_size[j + 1])
        center_num = math.exp((math.log(table['D Lower'][index_number - 1]) + math.log(table['D Upper'][index_number - 1])) / 2)
        table['D Center'].append(center_num)
        table['Volume Fraction'].append(curr_grain_vals[j + 1] - curr_grain_vals[j])
        class_number += 1
        index_number += 1
```

Using the information in the table, we can now find values like the volume fraction for each of the grains.

```bash
for i in range(3): #this is to get lists of the different vol_fracs
    name = grain_types_names[i]
    vol_fractions_grains[name] = []
    for j in range(num_of_sediments):
        vol_fractions_grains[name].append(vol_frac[table_counter])
        table_counter += 1
```

Find the combined GSD value for all of the sediments in the orignal GSD is a vlue of interest as this helps to give insight into flocculation. Below is the code on how using the volume fractions for each of the original sediment types and the information from the table can lead to this.

```bash
x_axis = np.arange(num_of_sediments)
y_axis_combined_GSD = []
for i in range(num_of_sediments):
    val_of_y = (proportion_s * vol_fractions_grains[grain_types_names[0]][i]) + proportion_k * vol_fractions_grains[grain_types_names[1]][i] + proportion_m * vol_fractions_grains[grain_types_names[2]][i]
    num = val_of_y / (math.log(table['D Upper'][i]) - math.log(table['D Lower'][i]))
    y_axis_combined_GSD.append(num)
total_GSD_value = sum(y_axis_combined_GSD)
for i in range(num_of_sediments):
    normalized_GSD = y_axis_combined_GSD[i] / total_GSD_value
    y_axis_combined_GSD[i] = normalized_GSD
```

Plotting the GSD results 

```bash
plt.xlabel('Grain Size (um)')
plt.ylabel('PDF')
plt.title('Different Grain Size Distribution Plot')
plt.xscale("log")
plt.plot(x_axis, y_axis_combined_GSD, label = "Combined GSD")
```

Exporting the results into a CSV file

```bash
keys = list(table.keys())
list_of_GSD_dict = []
for i in range(len(sediments)):
    new_dict = {}
    new_dict[keys[0]] = table[keys[0]][i]
    new_dict[keys[1]] = table[keys[1]][i]
    new_dict[keys[2]] = table[keys[2]][i]
    new_dict[keys[3]] = table[keys[3]][i]
    new_dict[keys[4]] = table[keys[4]][i]
    new_dict[keys[5]] = table[keys[5]][i]
    list_of_GSD_dict.append(new_dict)


with open('table_results.csv', 'w', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = keys)
    writer.writeheader()
    writer.writerows(list_of_GSD_dict)
```

Now that we have found the combined GSD values, the code below is plotting the individual GSD values.

```bash
individual_sediment_pdf = {}
for name in list(vol_fractions_grains.keys()):
   individual_sediment_pdf[name] = []
   proportion = dict_of_proportions[name]
   for i in range(len(vol_fractions_grains[name])):
       f = vol_fractions_grains[name][i]
       num = f / (math.log(table['D Upper'][i]) - math.log(table['D Lower'][i]))
       individual_sediment_pdf[name].append(num)
   sum_of_vals = sum(individual_sediment_pdf[name])
   for i in range(len(individual_sediment_pdf[name])):
       individual_sediment_pdf[name][i] = individual_sediment_pdf[name][i] / sum_of_vals

x_axis = np.arange(num_of_sediments)
for name_sediment, y_axis in individual_sediment_pdf.items():
    plt.plot(x_axis, y_axis, label = "GSD of " + str(name_sediment))

plt.legend()
plt.show()
```

Results of the combined vs individual GSD

![COMBINED_INDIVIDUAL_GSD](https://github.com/bayush11/Caltech_SURF_2023/assets/70395352/b715e0b2-5827-41b4-89c1-f89691841e7b)

# CDF Distribution
The CDF distribution code that I will show displays the CDF of the GSD versus the PSD data. The CDF is important as this will help us understand whether flocculation is occurring or not.

```bash
plt.close()
df = pd.read_csv('results.csv')
diameters = list(df.loc[:,"EquivDiameter"])
diameters = [float(i) * 0.81 for i in diameters]
CDF_for_GSD = []
for i in range(len(y_axis_combined_GSD)):
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
    if value not in CDF_for_PSD:
        CDF_for_PSD.append(value)
```

Plotting the results of the CDF.

```bash
x_axis_1 = list(np.arange(len(CDF_for_PSD)))
x_axis_2 = list(np.arange(len(CDF_for_GSD)))
plt.plot(x_axis_1, CDF_for_PSD, label = "PSD")
plt.plot(x_axis_2, CDF_for_GSD, label = "GSD")
plt.xscale("log")
plt.title('PSD vs GSD')
plt.xlabel('Diameter (um)')
plt.ylabel('CDF')
plt.legend()
plt.show()
```

Results of the CDF plotted between the PSD and the GSD.

![CDF](https://github.com/bayush11/Caltech_SURF_2023/assets/70395352/6418c8b2-ee1e-4351-a011-ab2019e0600e)

Below is a line of code used to gain information regarding the PSD. You will find the PSD plotted in a histogram while the GSD is in a line graph. That is because the PSD data is finite, so the information must be represented in a finite way.

```bash
(n, bins, patches) = plt.hist(medians_sample_1, density = True, bins= 15, label='PSD') #used to get info about PSD
```

The code below is used to plot the PDF of the GSD and PSD distributions.

Within this code, the GSD is normalized, and values that are lower than the smallest PSD value are cut off. The reason the cutoff occurs is that if there is a GSD value that is lower than the lowest PSD value, then that information won't be able to tell us trends on flocculation, so taking it away from our distribution is the smartest thing to do.

```bash
min_diameter = min(medians_sample_1)
y_axis_GSD = []
x_axis_2 = []
curr_diameters = table['D Center'][:100]
for i in range(len(curr_diameters)):
    if curr_diameters[i] >= min_diameter:
        y_axis_GSD.append(y_axis_combined_GSD[i])
        x_axis_2.append(curr_diameters[i])

sum = sum(y_axis_GSD)
for i in range(len(y_axis_GSD)):
    y_axis_GSD[i] = y_axis_GSD[i] / sum

plt.plot(x_axis_2, y_axis_GSD, label = "GSD")

plt.legend(loc='upper right')
plt.xscale("log")
plt.title('PSD vs GSD')
plt.xlabel('Diameter (um)')
plt.ylabel('PDF')
plt.legend()
plt.show()
```

Results of the PDF of the PSD vs GSD.

![PDF](https://github.com/bayush11/Caltech_SURF_2023/assets/70395352/81c9ff6c-4b5e-46f6-b908-e61462fd03a4)

Conducting statistical tests allows us to gather information that can indirectly help us see flocculation patterns. The line below helps to complete a Kolmogorov–Smirnov test and compares the PSD and GSD data.

# TODO

Make sure to fully explain the main.py file (aka the 16, 50, 80%)

Add the high level step by step for the training and testing process (e.g. do so and so and please refer to this link to help you with that)
Add the evniornment you are working in(google colab, python, matlab, etc).
reorder the steps to make sense
make sure to explain other .py files instead of mainly just table_creator.py
explain the code better and make it more organized
put comments at the top of each script to explain the script itself 
put comments in the lines as well 
delete commeneted code
add some google colab cells to the machine learning portion to explain the key points to use google colab and HOW to use google colab
images have to be jpg insetad of tif PUT THIS IN IMPORTANT
dont pause the training in coolavb 
add your tracker and detection models to github
svae the weights in the external hard drive
comment in the config files and explain what the variables are, why you might need to change it.
ake out any unnecessary code

```bash
print(ks_2samp(medians_sample_1, y_axis_GSD))
```

**ONCE AGAIN, ALL OF THIS INFORMATION IS IN THE table_creator.py FILE**

# Machine Learning and Tracking

Here I will provide the links and information that I found during the machine-learning portion of my project

Firstly, here are the links that I have primarily been using for my object tracking.

1. Video that describes how to track objects: https://www.youtube.com/watch?v=FuvQ8Melz1o
2. Video that describes how to create a custom object detector (important as this detector will be used to track custom objects): https://www.youtube.com/watch?v=nOIVxi5yurE
3. Video that describes how to create the custom dataset and train a custom detection model: https://www.youtube.com/watch?v=mmj3nxGT2YQ

All of these videos have links to GitHub that need to be accessed in order to complete the tracking.

# Training

The tracking can be classified into two sets: training the machine learning model and then using the trained model to track/detect objects. The link that I gave that shows how to make your own data/model will provide you links to not only make your own custom dataset but also how to train your machine learning model. Use this to create your model properly.

**NOTE: The training is done in google colab. After an hour or two, computing power will run out and ask you to buy stronger power for 10$ a month. DON'T DO THIS. Instead, switch between Google accounts and keep running your model through there. At one point you will exhaust your resources for the day. The computing power resets the next day**

# Detecting and Tracking

After your model is able to detect objects properly, use the first two links provided to use your trained model to track and train objects. Some area for potential error is that you are inputting the wrong number of classes or that you are using the wrong model. Please look up your errors online for this as there are plenty of resources that helped me to solve my issues.
