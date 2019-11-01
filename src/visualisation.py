import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

import imp
import innvestigate
import innvestigate.utils as iutils

imgnetutils = imp.load_source("utils_imagenet", "/content/drive/My Drive/src/utils_imagenet.py")

from sklearn.metrics import confusion_matrix

def visualise_images(samples_normal, samples_malaria):
  
  plt.figure(figsize=(15,15))
  all_samples = samples_normal + samples_malaria

  # Plot 6 random samples
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.subplots_adjust(bottom=0.4, top=0.9, hspace=0.2)
    k = np.random.choice(len(all_samples))
    label = 'Uninfected' if k <= (0.5*len(all_samples)-1) else 'Parasitized'
    plt.xlabel('True label : {}'.format(label),fontsize=15)
    plt.imshow(all_samples[k].squeeze(), cmap=plt.get_cmap('gray'), interpolation='nearest')


def plot_augmented_images(train_gen,batch_index,sample_index):
  ''' Inputs a train_generator, batch_index and sample_index,
      returns subplots of augmented images of that sample'''
  
  augmented_images = [train_gen[batch_index][0][sample_index] for i in range(4)]   
  fig, axes = plt.subplots(1, 4, figsize=(10,10))
  axes = axes.flatten()
    
  for img, ax in zip(augmented_images, axes):
    ax.imshow(img)
  plt.tight_layout()
  plt.show()


def get_confusion_matrix(model,test_gen):
  ''' model : A trained model
      validation_generator : A generator containing the validation samples
      outputs : 1. A plot of confusion matrix of true positives/ negatives and false positives/ negatives
                2. Arrays of true positive/ negatives, false positive/negatives'''

  # Get true labels and predictions
  labels = test_gen.labels
  predictions = model.predict_generator(test_gen)
  y_pred = np.argmax(predictions, axis=1)
  
  # Get the indices of right/ wrong predictions
  true_pos_idx, true_neg_idx, false_pos_idx, false_neg_idx = [], [], [], []
    
  batches = test_gen[0][0]

  for i in range(len(y_pred)):
    if labels[i] == 1 and y_pred[i] == 1:
      true_pos_idx.append(i)
    elif labels[i] == 0 and y_pred[i] == 0:
      true_neg_idx.append(i)
    elif labels[i] == 0 and y_pred[i] == 1:
      false_pos_idx.append(i)
    elif labels[i] == 1 and y_pred[i] == 0:
      false_neg_idx.append(i)
  
  # Join all the batches together
  batches = test_gen[0][0]
  for i in range(1,len(test_gen)):
    batches = np.concatenate((batches,test_gen[i][0]))
  
  # Change this part
  # Get the (96 x 96 x 3) arrays for each of the cases
  tp_arrays = batches[[true_pos_idx],:,:,:]
  tn_arrays = batches[[true_neg_idx],:,:,:]
  fp_arrays = batches[[false_pos_idx],:,:,:]
  fn_arrays = batches[[false_neg_idx],:,:,:]

  # Plot Heatmap
  df_cm = pd.DataFrame(confusion_matrix(labels, y_pred), index=[0,1], columns=[0,1])
  plt.figure(figsize=(10,10))
  heatmap = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.ylim(len(df_cm),-0.1)
  plt.xlabel('Predicted label')

  plt.show()  

  return tp_arrays, tn_arrays, fp_arrays, fn_arrays


def show_explanation(image_sample,analyzers,analyzer_names,methods):

  counter = 1
  plt.figure(figsize=(17.5,17.5))

  plt.subplot(1,4,counter)
  plt.xlabel('Original Image')
  plt.imshow(image_sample.squeeze())


  for i,analyzer in enumerate(analyzers):
    a = analyzer.analyze(image_sample)
    a = imgnetutils.postprocess(a, "BGRtoRGB",channels_first=False)
    a = methods[i][2](a)
  
    counter += 1
  
    plt.subplot(1,4,counter)
    plt.xlabel(analyzer_names[i])
    plt.imshow(a[0],interpolation='none')


