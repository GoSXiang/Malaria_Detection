import numpy as np 
import matplotlib.pyplot as plt

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



