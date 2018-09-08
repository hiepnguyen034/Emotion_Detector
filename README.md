# Emotion_Detector
Detect facial expression using opencv and linear algebra

The model detects if the person in the image is sad, neutral, happy, or surprised

# Dependencies

`tensorflow, opencv`

# Accuracy

93.8% on the training set and ~83% on my own images. This could be improved by editing the new images so that they look similar to the images in the train and dev sets (i.e. images should come from the same distribution)

# Computation graph

![computation_graph_l2](https://user-images.githubusercontent.com/29159878/45193868-5b942a80-b21e-11e8-8efe-e9062936fe0b.png)

# To test with specific image(s)

Clone the repos-> Open the ipynb and run every cell -> Insert path to image that needs to be tested -> run the function `test_specific_image(PATH)`. Input must be an image of one person, preferably a selfie with an empty background

