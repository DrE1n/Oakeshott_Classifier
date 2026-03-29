1. Purpose:
	This project implements a hierarchical AI image classification system for identifying medieval sword types according to the Oakeshott typology — a scholarly classification system which covers European swords from the end of early Middle Ages until Renaissance (roughly 10-16 centuries). A coarse ResNet50 classifier identifies the main sword type, followed by specialized fine classifiers for types with children subtypes.

2. Structure:
	In this folder I attach the files I used to organize the dataset, train it and visualize the performance with Streamlit. The confusion matrix and the training curves images are also here.

3. Models: 
	As for the models, here is the link to the Hugging Face: https://huggingface.co/Palatir/oakeshott_classifier/tree/main
The approach of using different models is to let the coarse model to predict the parent type of the sword(X, XI, XII etc) and then, provided that the predicted type has children subtypes(Xa, XVIIIa, XVIIIb etc), apply fine models which are trained only to distinguish the swords within the parent domain.

4. Application:
	To use it, just download the inference.py file, the models and specify the MODELS_DIR path to "oakeshott_classifier" in the inference file. Then, run the inference file from the terminal to start the streamlit app. To classify the sword, just drag its image to the classifier.

5. Dataset: 
	The dataset has been assembled using the open source images of authentic artifacts from museum collections, scientific works and replicas from modern sword manufacturers. I also used rendered images of the open source 3d sword models to boost the underrepresented sword types data.

6. Results:
	At the moment the classifier is still a bit far from perfect, it still misses some quite distinctive sword images. However, with an average accuracy of about 55% for the coarse classifier, it handles it's task better than I expected before the first launch. I will try improve it's performance in the future. As for the exact confusion matrices and training curves, please see the results folder.

7. Future work:
	The main thing to improve is to expand the dataset, especially for rare and controversial types that the coarse and fine models often miss. The general goal would be to improve each parent type prediction accuracy to at least 50-60% so that there are less misses. Secondly, I would like to introduce the types which are currently missing in this classifier. Unfortunately, I had to skip few rare classes(type XIIIb and type XXII) as it was impossible for me at this stage to find enough data on them to build a somewhat useful classifier. That said, I will keep updating the models on Hugging Face if I find more images to upgrade the models performance for the current and missing swords. 

