from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def get_data(path,limit):
  
  ''' Given a path, this function returns four lists created from the path -
  
     eg. 'path = 'train_path'' returns data in the NORMAL / MALARIA folders
     corresponding to this path
     
     limit is imposed to save memory, this means the no. of data obtained is only
     up to that limit
  
  
     samples_normal : all samples for the 'NORMAL' images
     filenames_normal : filenames corresponding to samples_normal
      
     samples_malaria : all samples for the 'MALARIA' images
     filenames_malaria: filenames corresponding to samples_malaria
  '''

  samples_normal, samples_malaria = [], []
  filenames_normal, filenames_malaria = [], []

  # Get data for normal images
  for filename in os.listdir(path + 'Uninfected')[:limit]:
  
    if filename != '.DS_Store':
      image = Image.open(os.path.join(path + 'Uninfected', filename))
      imarray = np.array(image)
      samples_normal.append(imarray)
      filenames_normal.append(filename)
  
  print('Files in first {} images of {} appended!'.format(limit,path + 'Uninfected'))

  
  # Get data for pneumonia images
  for filename in os.listdir(path + 'Parasitized')[:limit]:
  
    if filename != '.DS_Store':
      image = Image.open(os.path.join(path + 'Parasitized', filename))
      imarray = np.array(image)
      samples_malaria.append(imarray)
      filenames_malaria.append(filename)
  
  print('Files in first {} images of {} appended!'.format(limit,path + 'Parasitized'))


  return samples_normal, samples_malaria, filenames_normal, filenames_malaria



def fit_into_generators(train_df,valid_df,test_df,train_imgen,valid_imgen,batch_size=32,shape=(96,96)):
  ''' Takes in 3 dataframes and returns generators : train, test and validation '''

  train_generator = train_imgen.flow_from_dataframe(
  dataframe = train_df,
  directory='Combined/',
  x_col='train_combined',
  y_col='labels',
  has_ext=True,
  batch_size=batch_size,
  seed=2018,
  shuffle=True,
  class_mode='categorical',
  #classes=['0','1'],
  target_size=shape)

  valid_generator = valid_imgen.flow_from_dataframe(
  dataframe = valid_df,
  directory='Combined/',
  x_col='val_combined',
  y_col='labels',
  has_ext=True,
  batch_size=batch_size,
  seed=2018,
  shuffle=False,
  class_mode='categorical',
  #classes=['0','1'],
  target_size=shape)
    
  test_generator = valid_imgen.flow_from_dataframe(
  dataframe = test_df,
  directory='Combined/',
  x_col='test_combined',
  y_col='labels',
  has_ext=True,
  batch_size=batch_size,
  seed=2018,
  shuffle=False,
  class_mode='categorical',
  #classes=['0','1'],
  target_size=shape
  )
    
  return train_generator, valid_generator, test_generator
