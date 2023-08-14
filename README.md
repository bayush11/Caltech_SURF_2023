# SURF CALTECH 2023 README

# Background

Below I will be discussing the particle tracking project that was being done over a 10-week course at Caltech summer 2023. I will give insight onto any results that I have gotten as well as provide code to help future readers of this document. The data of the grains and particles that will be discussed are available due to the work of Kim and Kyd who conducted particle experiments. Some important information to know is that GSD stands for grain size distribution and represents the sizes of the grains at the beginning of the experiments, while PSD is the particle size distribution and represents the particle sizes throughout the experiment.

# Imaging Data 

The imaging data was processed through the Keyvani and Strom (2013) workflow. This is the paper link: https://doi.org/10.1016/j.cageo.2012.08.018. I have additionally linked

Below are the different phasese that the images have gone through in the Keyvani and Strom (2013) workflow. When processing a time-series, one should get folders of these three image types plus a binarized labeled image.

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
Below is the code that plots the frame by the diameter of the flocs. When using the Keyvani and Strom (2013) workflow, one will recieved a csv file that has the diameters of every particle tracked in the binarized images. This code will take that csv and find the median diameter per frame. **THIS IS IN THE MAIN.PY FILE.**

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

# Table Creation

The table that I will be discussing is a file that I created to list the many important features regarding grain size distribution information. This table will be used throughout the rest of the code as the information is relevant.

**FURTHER INFORMATION CAN BE FOUND IN THE table_creator.py FILE**

```bash
table = {}
table['sediment'] = []
table['class number'] = []
table['D Lower'] = []
table['D Upper'] = []
table['D Center'] = []
table['Volume Fraction'] = []
index_number = 1
for i in range(3):
    class_number = 1
    curr_grain_type = grain_types_names[i]
    curr_grain_vals = grain_types[curr_grain_type]
    for j in range(len(curr_grain_vals) - 1): #n-1 iterations as we are taking the class
        table['sediment'].append(curr_grain_type)
        table['class number'].append(class_number)
        table['D Lower'].append(grain_size[j])
        table['D Upper'].append(grain_size[j + 1])
        center_num = math.exp((math.log(table['D Lower'][index_number - 1]) + math.log(table['D Upper'][index_number - 1])) / 2)
        table['D Center'].append(center_num)
        table['Volume Fraction'].append(curr_grain_vals[j + 1] - curr_grain_vals[j])
        class_number += 1
        index_number += 1
```

# PDF Distribution

The PDF distribution code that I will show displays the PDF of the GSD versus the PSD data.

