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

1. Add the high level step by step for the training and testing process (e.g. do so and so and please refer to this link to help you with that)
2. Add the evniornment you are working in(google colab, python, matlab, etc).
4. make sure to explain other .py files instead of mainly just table_creator.py
5. explain the code better and make it more organized
6. put comments at the top of each script to explain the script itself 
7. put comments in the lines as well 
8. delete commeneted code
1. add some google colab cells to the machine learning portion to explain the key points to use google colab and HOW to use google colab
1. images have to be jpg insetad of tif PUT THIS IN IMPORTANT
1. dont pause the training in coolavb 
1. add your tracker and detection models to github
1. svae the weights in the external hard drive
1. comment in the config files and explain what the variables are, why you might need to change it.
1. ake out any unnecessary code
1. specifically describe how to train the model in the yolov4-deepsort video
1. DIFFERENTIATE between tensorflow and google cloud
1. link the githubs and the google colab
1. talk abt info flag and how this can be used to get the settling velocity
1. write documentation regarding the different detectors I have and what they essentially
1. add documentation regarding the matlab parameters and discuss how I got the videos, how much of the video did you use and the sampling rate
1. Number of frames used to train each of the models
1. add info regarding the config files per 
1. potential ways to make model better
1. write up how to run the object detector
2. The training is done in google colab. After an hour or two, computing power will run out and ask you to buy stronger power for 10$ a month. DON'T DO THIS. Instead, switch between Google accounts and keep running your model through there. At one point you will exhaust your resources for the day. The computing power resets the next day**
3. Make sure to input images when you are saying you will
4. I didnt expalin too much abt the labeling, is that chill
5. When talking about the labeling, you dont every discuss how YOU DID IT. TALK ABT THAT.

```bash
print(ks_2samp(medians_sample_1, y_axis_GSD))
```

**ONCE AGAIN, ALL OF THIS INFORMATION IS IN THE table_creator.py FILE**

# Machine Learning and Tracking

Here I will provide the links and information that I found during the machine-learning portion of my project

Firstly, here are the links that I have primarily been using for my object tracking.

If you want to jump ahead and reference my step-by-step instructions later on, consult these videos as they are very helpful in guiding you.

1. Video that describes how to create the custom dataset and train a custom detection model: https://www.youtube.com/watch?v=mmj3nxGT2YQ
2. Video that describes how to create a custom object detector (important as this detector will be used to track custom objects): https://www.youtube.com/watch?v=nOIVxi5yurE
3. Video that describes how to track objects: https://www.youtube.com/watch?v=FuvQ8Melz1o


All of these videos have links to GitHub that need to be accessed in order to complete the tracking.

# Tracking Model Creation

There are two large steps that are required in creating a proper machine learning model to track particle. The **first** step is to develop your model through creating labels from images (1) and then using those labels for training your machine learning model (2). This step is nessecary as this will allow you to develop the model that will be used for detecting and tracking future objects. The **second** step in the process is to use this trained model to detect (3) and track objects(4).

The numbers inside of the paraenthesis represent the step down below that will go deeper into how to complete this.

I will cover a high level step-by-step process on how to complete these two steps.

# Step 1: Developing Labels 

**Label Format: YOLOV4**

Lets dive into how to first get your labeled images. This is important as in computer vision, the main goal that you want is for computer to be able to recognize shapes and objects on a screen. With that in mind though, how can a computer even being to understand this? The answer is through labeled images. I have provided an example of an image, and then a image with a label to help give some context. **PUT AN IMAGE BELOW**

A labeled image tells the computer **WHERE** exactly the object is and **WHAT** the objrect is. Doing this, the computer will know how exactly to classifiy future instances of a similar object. How can we exactly get these labels now? There are two ways this can be done. The first way is to autmoatically get the labels from the images by using online software and code. Here is a link to a way to download images and their labels online (https://www.youtube.com/watch?v=_4A9inxGqRM). The other way is to manually draw a bounding box around each of the objects within your image and to store the labels through there.

I personally recommend automating this process. At the start of this project, I orignally used the manual way as I already had custom images (the autmoated way above only works if you use google images); with that in mind, I found a way to automatically get bounding boxes for the particles within each image using the Keyvani and Strom (2013) workflow (https://doi.org/10.1016/j.cageo.2012.08.018).

I want to sasy again that it is VERY VERY IMPORTANT that you find a way to automate this process. The reason being is that the more labels and images you can have, the more accurate your object detector/tracker can be. Don't try to compromise this step.

I want to quickly discuss the labels themselves. Each label represents the bounding boxes around the objects that you are trying to detect. The exact format is not extremerly important as there are specific ways that the labels are converted; the overall message though is that they are each of the bounding boxes around objects in an image. These lables will be what we are using to train the machine learning model as again, the labels give the computer a location to actually see the images.

These labels are stored in a txt with each txt being filled with all of the labels for every object within an image. So for example, an image with 8 bounding boxes will have 8 labels and 8 lines in the txt. 

# Step 2: Training the Model

Now that we have our labels and images in the correct format, it is time to train the machine learning model. One quick note is that the way that the folder that contains your training data and the other folder that has your testing data should be set up in a way that each image should be next to its label. So an example is that for traning data, the folder contents could go like this: img1.jpg, img1.txt, img2.jpg, img2.txt, etc. Both your training and testing folders will just have the images and their corresponding labels. More information can be found by referencing my obj and test folders in the repo.

Now onto the model! We will be training the model through a google colab (https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing). While I wont go into the exact details on how to train your model since they can be found in this link (https://www.youtube.com/watch?v=mmj3nxGT2YQ), I do want to go over abstractly what this step accomplishes and why is it useful. For one, when completeing this step, this will be done all throughout google colab. 

The way that colab will train your model is through referencing a few files: a configuration file, your training and testing .zip files containin your images and labels in the correct order, a obj.names file that has the class names you will be training on, a obj.data file that will have path information to things like images and labels, a generate_test.py and a generate_train.py file to locally obtain the training and testing images, and finally a backup folder. The configuration file is extremly useful as this file will denote how exactly this model will be trained. I have made comments on important variables within the saved config files in the repository. With that being said, some of the important variables to just be aware are the height and width (allow the computer to see a certain amount of a image), the max_batches (allow for the max bumber of iteratiosn the model will train on), classes, filters, batches and subdivisions. These are all parameters that are able to be changed depending on the model you are creating. I typcailly advise not to change them though as they do not vary a ton from our application standpoint.

One other very very important parameter is the backup folder. The backup folder will contain all of the weights that are recieved from the training process. These weights are what makes our trained model different from the next one. They are specific to the training and testing data given, so keep a VERY close eye on your trained weights.The generate_test.py and a generate_train.py files stay the same from application to application. The obj.names and obj.data will change only if your classes change.

In using these files, the model will now train to help more accurately detect and track certain objects. Somethings to take note off are that your model should have an average loss below one in a ideal situation. It isnt terrible if its above one but the detections and tracking will suffer as this number keeps going up. 

Throughout the google colab, you will be told how to get these resources and to store them in your yolov4 folder which should be located in your google drive. IT IS IMPORTANT THAT YOU CREATE YOUR YOLOV4 FOLDER AS EMPTY FIRST. After doing this, the steps in the video will guide you on how to fill the yolov4 folder with the proper files. Train your model for a few hours, you typically dont need to go above 4000 iterations. Once that is done, checn the mAP value (Mean Average Precision) or in other words, the accuracy of the model. Then choose if you want to change some parameters to make your training better or to move on. If you are satisfied then congrats, you made your model that we will use to detect and track!

# Step 3: Detecting Objects

**NOTE**: This will involve using a TensorFlow training process which was different from the previous Cloud format of training.a

Now that we have our trained model, we can use them for two applications : detecting objects in images/videos or tracking objects in videos. In this subsection, we will focus on detecting objects. The first step is to reference this YouTube link (https://www.youtube.com/watch?v=nOIVxi5yurE) that will help you step by step on what to do to properly detect objects in your images. With tat being said, I will still cover the high level process on what needs to be done.

The first step is to clone this github repository (https://github.com/theAIGuysCode/tensorflow-yolov4-tflite) and to install the proper requirments from the requirements.txt. 

**IMPORTANT**: In the current txt, there is a line that says tensorflow==2.3.0rc0. Change this to tensorflow==2.3.0 and this will solve any errors you have.

After this, you will need to use this command to properly train your model in TensorFlow: python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4. One might ask, didnt we just train the model? Why again? The reason being is that we trained the model in cloud in google colab. This allowed us to gain the weights associated with our machine learning model. We can now use these weights to proerly train the model in TensorFlow which makes detecting/tracking objects much easier than in cloud.

Additiaonlly, for any image of video that you want to detect, enter into data/images or data/video to store your images and videos. This will be nseecary in the near future.

With that in mind, the custom.weights file needs to be the weights that you recieved from the training in cloud proces. This ensures that the proper TensorFlow model can be accrautely trained to detect your specific image inputs. After this is done, let the model run for a bit as it takes a few minutes to train.

After the model is trained, use this command to see if there are proper bounding boxes around the objects of interest: python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car.jpg. Replace the "car.jpg" with the image that you want to detect from your images folder. Run this command and see your output!

Similarly, if you want to imitate this process for a video isntead, use this command python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/cars.mp4 --output ./detections/results.avi. Replace the cars.mp4 with the proper video you have referenced and check your detections.

If the output was not what you wanted, then potentially your model needs more training or the image is very unclear. Try different things to see what allows for better accuracy. The output bounding box has a small number that is <= 1 and this represents the models confidence that the bounding box has accuractely tracked an objects of interst.

# Step 4: Tracking Objects
