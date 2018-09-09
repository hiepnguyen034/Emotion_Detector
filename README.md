# Emotion_Detector
Detect facial expression using opencv and linear algebra

The model detects if the person in the image is sad, neutral, happy, or surprised

# Dependencies

`tensorflow, opencv`

# Accuracy

93.8% on the training set and ~83% on my own images. This could be improved by editing the new images so that they look similar to the images in the train and dev sets (i.e. images should come from the same distribution)

# Computation graph

![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run](https://user-images.githubusercontent.com/29159878/45260314-451ed800-b3b2-11e8-8ed1-18e083e2f6c6.png)


# To test with specific image(s)

Clone the repos-> Open the notebook `customized_tests.ipynb` and run all the cells before #CUSTOMIZED TEST CASES START FROM HERE# cell -> Insert path to the image that needs to be tested -> run the function `test_specific_image(PATH)`. Input must be an image of one person, preferably a selfie with an empty background

