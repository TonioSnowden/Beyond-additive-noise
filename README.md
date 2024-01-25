# Beyond Additive Noise
Structure of a very unstructured repo:

- Notebook augmentations.py show how the text images are modified and distored. The augmentations were done careffuly to not disrupt the visibilty of the text and to simulate real world scenarios found in phone camera images.

- https://docs.google.com/spreadsheets/d/1C3Z2f67skPnN6IdXlDt9Fyg6Zwl43u0PFoHk-MaKAc8/edit?usp=sharing:
  results for testing the sota on the augmentations

- img_ground_truth.csv, img_ground_truh_train.csv: these files are a csv mapping from image name and its' label, from the IIIT5k dataset, generated using matlab. It was used for initial exploration of the task

- crnn-kaggle.ipynb: Kaggle provides many datasets (samples) that are useful and public. Send Your Kaggle user account and i will share what is needed.

  **The model is the SoTa CRNN.**

  >
  >There are also useful cells for datasetBuilding and preprocessing of the pairs (image, text) input.
  >
  >-> Next is to back them up as TFRecords to save time,  and share it. 
  >
  >This can be better optimized to gain a bit of time computation. (depends if we'll keep working with it)

- "one_epoch.h5": (If you want the model's weight you can ask me by mail : antoine.munier@epfl.ch) These are the loaded weights of one training epoch of this model. We can build our model with these pre-saved weights and continue training. 
  Training (and preprocessing etc) it took 5 hours on Kaggle GPU T4x2.  Given the complexity of the model and the size of the data, this was expected. 
  `  31356/Unknown - 18123s 577ms/step - loss: 1.8949 - sequence_accuracy: 0.8107`: output of model.fit()

- "DCGAN.ipynb": initial code for GAN that tries to generate augmented dataset

- "TPS_ResNet_BiLSTM_Attn_.ipynb": more promising Sota
