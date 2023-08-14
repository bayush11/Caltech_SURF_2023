# SURF CALTECH 2023 README

# Background

Below I will be discussing the particle tracking project that was being done over a 10 week course at Caltech summer 2023. I will give insight onto any results that I have gotten as well as providing code to help the future readers of this document.

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

***Before going into the code, this code snippet below shows how to convert from the frame number to time. The FPS is 60, so this is just a multiplication factor that is needed.***
'''
for i in range(len(imgNumPlt)):
    imgNumPlt[i] *= 60 #60 FPS, converting to time
'''
Below is the code that plots the frame by the diameter of the flocs. When using the Keyvani and Strom (2013) workflow, one will recieved a csv file that has the diameters of every particle tracked in the binarized images. This code will take that csv and find the median diameter per frame. **THIS IS IN THE MAIN.PY FILE.**

This code snippet helps to get the number of frames there are in the csv file.

for i in range(len(imgNum)):
    if imgNum[i] not in imgNumPlt:
        imgNumPlt.append(imgNum[i])

Thsi is where using the unique numbers, one finds the median diameter per frame.

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

This is now plotting the results.

axis[1].plot(imgNumPlt, medians_sample_1)
