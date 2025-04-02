# Dog Breed Classifier

This project is a Dog Breed Classifier, deployed using Streamlit Cloud. 

Model designed for educational purposes.
## Dataset

The [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) is used. This dataset contains 120 different dog breeds.
## Dependencies

The following libraries are used (check `requirements.txt`):
- `streamlit`
- `torch`
- `torchvision`
- `pillow`
- `scikit-learn`
- `tqdm`

## Model

A pretrained ResNet18 model was fine-tuned by modifying the final layers to better capture breed-specific features.

Other details are as follows:
- **Loss Function:** Cross Entropy Loss
- **Optimizer:** Adam  
- **Learning Rate Scheduler:** Gradual learning rate decay to accelerate convergence
- **Validation Metrics:** Top-1 and Top-5 accuracy
- **Early Stopping:** Triggered after 10 epochs without improvement in Top-1 accuracy  

Training was performed using the `dog-breed-classifier.ipynb` notebook on Kaggle's GPU accelerator due to limited local resources.

The model achieved a maximum Top-1 validation accuracy of 76.21%. I'd like to enhance this performance in future improvements.

## To implement

I'd like to add:

- Adjust parameters/change model for higher accuracy.
- Add more data augmentation so the model can stay robust to different angles or lightings.
- Explainability features:
    - Add GradCAM
    - Add brief textual explanation of the features that made the model reach its conclusion and a comparison to common features of that breed.
    - If confidence is low (<60%), make the model explain why it's not so confident and what is the second most-likely breed.
- Add more recognition classes:
    - Perhaps recognize if the uploaded image is not a dog...?
    - Recognize if the dog might be a mix of breeds. 
- Add age prediction feature