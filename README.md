# Introduction

In this modern society, explanability of well built AI models is an issue, owing to the 'black box' mechanics behind these well-performing models. For this project, the aim is to first build a few models that can predict whether a cell is 'Parasitized' or 'Unparasitized' by the Malaria disease with a high accuracy. Following which, we will look into what these 'good' models are learning, given an input image, such as which parts in the cell image contribute the most to a particular prediction.

## Structure

| Folder         | Description                                                                                                     |
| :---          | :---                                                                                                            |
|data        | Contains the train, test and validation set                               |
|notebooks       | Contains the google colab notebooks on training and analysis     |
|src    | Contains the python scripts required for the training and analysis          |
|weights     | Some sample weights stored in h5 format                      |

##  Evaluation Metrics

Here, the metrics used is accuracy, which refers to the proportion of correct predictions compared to the true classes (Parasitized/ Uninfected) by the model. In the training, the loss function used is the categorical crossentropy loss.

## Models used

The table below shows the models used for analysis. Firstly, a baseline model is built as a default reference. Secondly, transfer learning on the various pretrained Keras models was done using the Malaria dataset. All of the models were trained/ fine-tuned for 10 epochs. For all of the models that were used for transfer learning, most of the layers were frozen, with the exception of the last few layers and the additional layers.

| Model        | Description      | Accuracy                                                                                                   |
| :---          | :---            | :---                                                                                                 |
|Baseline        | Built using just a few convolutional layers | 95.5%                               |
|Finetuned ResNet50       | Added a few layers from the pretrained Keras ResNet50 (Transfer Learning)  | 94.9%
|Finetuned InceptionV3    | Added a few layers from the pretrained Keras InceptionV3 (Transfer Learning)      | 94.5%
|Finetuned XceptionV1     | Added a few layers from the pretrained Keras XceptionV1 | 95.8%                      

