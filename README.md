# An ensemble of modality-specific deep learning models for improved-TB-detection in chest radiographs

## Aim:

We propose a modality-specific knowledge transfer strategy to explore the possibilities of building models that learn generic features from a large-scale collection of CXRs and repurposing them to detect TB-like manifestations. We propose model ensembles by combining multiple models, expecting that it would lead to predictions that are better than any individual, constituent model. We study the combined benefits of multi-model knowledge transfer and ensemble learning toward improved TB detection in chest radiographs. 

## Prerequisites:

Anaconda3 >=-5.1.0-Windows-x86_64

Jupyter Notebook

Keras >= 2.2.4

Tensorflow-GPU >= 1.9.0

## Data collection:

The following publicly available CXR datasets are used in this study:

Pediatric pneumonia dataset: The dataset includes anterior-posterior (AP) CXRs of children from 1 to 5 years of age, collected as a part of routine patient care, from Guangzhou Women and Children’s Medical Center in Guangzhou, China. It includes 1,583 pediatric normal CXRs and 4,273 radiographs infected with bacterial and viral pneumonia. The dataset is curated by expert radiologists and screened to remove low-quality, unreadable radiographs. 

Radiological Society of North America (RSNA) pneumonia dataset: The dataset is hosted by the radiologists from RSNA and Society of Thoracic Radiology (STR) for the Kaggle pneumonia detection challenge toward predicting pneumonia in CXRs. It includes a total of 17833 abnormal and 8851 normal radiographs in DICOM format with a spatial resolution of 1024×1024 pixel dimensions and 8-bit depth.

Indiana Dataset: The dataset includes 2,378 abnormal and 1726 normal, posterior-anterior (PA) chest radiographs, collected from hospitals affiliated with the Indiana University School of Medicine, and archived at the National Library of Medicine (NLM) (OHSRP# 5357). The collection is made publicly available through the OpenI® search engine developed by NLM. 

Shenzhen Dataset: The dataset includes 336 TB-infected and 326 normal radiographs and the associated clinical readings, collected from the outpatient clinics of Shenzhen No.3 People’s Hospital, China. Table 1 shows the distribution of data across the train and test sets for the different dataset collections used in this study. 

## Preprocessing:

The following preprocessing steps are applied in common to these datasets: (a) median-filtering with a 3×3 window for edge preservation and noise removal; (b) resizing to 224×224 pixel resolution to reduce computational complexity and memory requirements; (c) rescaling to restrict the pixels in the range [0, 1]; and (d) normalization and standardization through mean subtraction and division by standard deviation to ensure similar distribution range for the extracted features. 

## Models and computational resources:

The performance of the following CNNs are evaluated toward the task of detecting TB in CXRs: (a) customized CNN; (b) VGG-16; (c) Inception-V3; (d) InceptionResNet-V2; (e) Xception; and (f) DenseNet-121. 

## Modality-specific knowledge transfer:

The overall process is described herewith: 

(a) Model A: The custom and pretrained models, otherwise called the base models, are trained on datasets including RSNA pneumonia, pediatric pneumonia, and Indiana collections, to learn the CXR domain-specific features and classify them into abnormal and normal categories. We randomly selected 10% of the training data toward validation. Callbacks and model checkpoints are used to investigate the performance of the models after each epoch. Learning rate is reduced whenever the validation accuracy ceased to improve. The retrained models with the best test classification accuracy are stored for further evaluation. 

#### The Jupyter notebook modelA_training.ipynb illustrates the process.

(b) Model B: The base models are trained and evaluated with the Shenzhen dataset collection, to categorize into TB-infected and normal classes. The models are evaluated through five-fold cross-validation to prevent overfitting and improve robustness and generalization. The retrained base models with the best model weights, giving the highest test classification accuracy for each cross-validated fold are stored for further evaluation.

#### The Jupyter notebook modelB_training.ipynb illustrates the process.

(c) Model C: Retrained models from Model A with CXR domain-specific knowledge is fine-tuned on Shenzhen dataset collection to categorize into TB-infected and normal classes. Embedding domain-specific knowledge is expected to improve model adaption to the target task. The models are evaluated through five-fold cross-validation.  The retrained models showing the best performance for each cross-validated fold are stored for further evaluation. With multi-model knowledge transfer, Model C is expected to demonstrate improved TB detection performance as compared to Model B.

#### The Jupyter notebook modelC_training.ipynb illustrates the process.

## Ensemble learning:

The predictive models from Model C are combined through majority voting, simple averaging, weighted averaging, and stacking to classify the CXRs into TB-infected and normal classes. We used a neural network as a meta-learner to learn from the predictions of the top-performing models from Model C. The layers in the base-learners are marked as not trainable so the weights are not updated when the stacking ensemble is trained. The outputs of the base-learners are concatenated. A hidden layer is defined to interpret these predictions to the meta-learner and an output layer to arrive at probabilistic predictions. 

#### The Jupyter notebook model_ensemble.ipynb illustrates the process.

## Performance metric evaluation:

The models in multi-model knowledge transfer and ensemble pipeline are evaluated in terms of the following performance metrics: (a) accuracy; (b) AUC; (c) mean squared error (MSE); (d) precision; (e) sensitivity; (f) specificity; (g) F-score; and (h) Matthews Correlation Coefficient (MCC). The models are trained and evaluated on a Windows system with Xeon CPU, 32GB RAM, NVIDIA 1080Ti GPU and CUDA/CUDNN for GPU acceleration. The models are configured in Python using Keras API with Tensorflow backend.

## Citation:

If you find these codes useful, kindly consider citing this publication: 

### S. Rajaraman and S. K. Antani, "Modality-Specific Deep Learning Model Ensembles Toward Improving TB Detection in Chest Radiographs," in IEEE Access, vol. 8, pp. 27318-27326, 2020, doi: 10.1109/ACCESS.2020.2971257.
