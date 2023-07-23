# HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification

Implemented the paper :- HybridSN_Exploring_3-D2-D_CNN_Feature_Hierarchy_for_Hyperspectral_Image_Classification (https://ieeexplore.ieee.org/document/8736016)

# HybridSN
 1- HybridSN is a hybrid CNN architecture that combines 3-D and 2-D CNNs to exploit the spatial and spectral information of hyperspectral images. 
 2- The proposed architecture first extracts spectral features using a 3-D CNN and then spatial features using a 2-D CNN. 
 3- The two streams are then combined to classify the hyperspectral image.


How to run the code :
 
1. Run the python file HybridSN_ML.py.
2. A pop up will be shown where you need to provide the required image data file provided in data folder. I have used Indian Pines and Salinas-A dataset.
3. Another pop up will be shown where you need to provide the required image ground truth file provided in data folder.
3. It will ask you for the no. of epoches provide suitable epochs, from our observation even 30 epochs give good result.
4. wait for some time it will show the final True and predicted images side by side.
5. After that it will show the plot of training accuracy vs epochs.
6. At last, it will show the whole classification Report for the given dataset.
