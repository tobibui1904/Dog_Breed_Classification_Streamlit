import torch
from PIL import Image
import requests
from torchvision.models import resnet50, ResNet50_Weights
import io
import cv2
import numpy as np
import streamlit as st
from albumentations import Compose, CenterCrop, Normalize
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
import plotly.express as px 
from streamlit_extras.no_default_selectbox import selectbox
from streamlit_extras.colored_header import colored_header
from streamlit_extras.streaming_write import write
import time

# Define the number of classes in your dataset
NUM_CLASSES = 120  # Replace with the number of your classes

# Load the model from the best checkpoint path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subset = ['Lhasa', 'beagle', 'Labrador_retriever', 'Norwich_terrier', 'Newfoundland', 'cairn', 'schipperke', 'Border_terrier', 'Mexican_hairless', 'Irish_setter', 'affenpinscher', 'basset', 'Norfolk_terrier', 'chow', 'Leonberg', 'toy_poodle', 'vizsla', 'kelpie', 'Brittany_spaniel', 'Pekinese', 'Weimaraner', 'Rottweiler', 'Ibizan_hound', 'groenendael', 'Bernese_mountain_dog', 'Boston_bull', 'collie', 'Tibetan_terrier', 'Airedale', 'toy_terrier', 'Saluki', 'Saint_Bernard', 'komondor', 'Kerry_blue_terrier', 'keeshond', 'Maltese_dog', 'black', 'Italian_greyhound', 'Pembroke', 'whippet', 'papillon', 'malinois', 'giant_schnauzer', 'Australian_terrier', 'golden_retriever', 'clumber', 'Great_Dane', 'Chesapeake_Bay_retriever', 'basenji', 'Welsh_springer_spaniel', 'Blenheim_spaniel', 'Japanese_spaniel', 'Tibetan_mastiff', 'curly', 'silky_terrier', 'Irish_wolfhound', 'Sussex_spaniel', 'flat', 'Bedlington_terrier', 'wire', 'EntleBucher', 'bluetick', 'briard', 'German_short', 'Lakeland_terrier', 'dhole', 'Cardigan', 'borzoi', 'French_bulldog', 'standard_poodle', 'Appenzeller', 'Border_collie', 'Pomeranian', 'kuvasz', 'miniature_pinscher', 'pug', 'bloodhound', 'Irish_terrier', 'American_Staffordshire_terrier', 'dingo', 'West_Highland_white_terrier', 'English_setter', 'Scotch_terrier', 'Bouvier_des_Flandres', 'Greater_Swiss_Mountain_dog', 'standard_schnauzer', 'malamute', 'miniature_poodle', 'Siberian_husky', 'soft', 'cocker_spaniel', 'Great_Pyrenees', 'Brabancon_griffon', 'Shetland_sheepdog', 'Doberman', 'Eskimo_dog', 'boxer', 'Old_English_sheepdog', 'Sealyham_terrier', 'bull_mastiff', 'Dandie_Dinmont', 'Walker_hound', 'miniature_schnauzer', 'Afghan_hound', 'Staffordshire_bullterrier', 'Scottish_deerhound', 'Chihuahua', 'Norwegian_elkhound', 'German_shepherd', 'Gordon_setter', 'redbone', 'otterhound', 'Samoyed', 'African_hunting_dog', 'English_foxhound', 'Yorkshire_terrier', 'Shih', 'Irish_water_spaniel', 'English_springer', 'Rhodesian_ridgeback']

colored_header(
    label="Dog Classification",
    description="Made by Tobi Bui",
    color_name="yellow-80",
)

if st.checkbox("Description"):
    description = "Welcome to my dog classification app. On the main page, the App tab on the sidebar, I used the best checkpoints that I trained before with the pretrained Resnet50 model for image classification. The accuracy after evaluation is around 80% while the best checkpoint's accuracy is around 99%. This model might not be completely true but please enjoy this little project I made. Thank you so much. I hope you like it"
    description_list = description.split(" ")
    def stream(description_list):
        for word in description_list:
            yield word + " "
            time.sleep(0.1)
    write(stream(description_list))

option = selectbox("Select an option", ["Link", "Local Image Upload"])
st.write('You selected:', option)

if option == None:
    st.warning("Please select one of the 2 options in the box", icon="⚠️")
    pass
else:
    upload = None
    image_url_state = False
    if option == "Local Image Upload":
        upload = st.file_uploader(label = "Picture Uploader", type= ['jpg', 'jpeg', 'png'])
        if upload == None:
            st.warning("Please choose the dog picture from your device", icon="⚠️")
        else:
            img = Image.open(upload)
            st.image(img, width= 700)
            img = np.array(img)

    elif option == "Link":
        # Check valid Image url
        def is_valid_image_url(url):
            try:
                response = requests.head(url)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type')
                    if content_type and content_type.startswith('image'):
                        return True
            except requests.RequestException:
                pass
            return False
        
        # Download and preprocess the image
        image_url = st.text_input("Please type the correct image link")
        if image_url:
            image_url_state = True
            if is_valid_image_url(image_url) == True:
                st.image(image_url, width=700)
                response = requests.get(image_url)
                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                img = np.array(img)
            else:
                st.warning("Retype the correct image link please", icon="⚠️")
                pass
        
    if (image_url_state == True and upload == None and is_valid_image_url(image_url) == True) or (image_url_state == False and upload != None):
        img = cv2.resize(img, (224, 224))
        augmented = Compose([ToTensorV2(Normalize)], p=1)(image = img)
        img_tensor = augmented['image']
        # img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        # Move the input tensor to the correct device
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device, dtype=torch.float32)

        # Use the model to make predictions
        with torch.no_grad():
            # Create and initialize the ResNet-50 model
            resnet50_model = resnet50(progress = True, weights=ResNet50_Weights.DEFAULT)

            # Modify the final fully connected layer for the number of classes
            num_features = resnet50_model.fc.in_features
            resnet50_model.fc = torch.nn.Linear(num_features, len(subset))
            resnet50_model.load_state_dict(torch.load('/home/nbui/Public/dog_breed/checkpoints/resnet50_checkpoint_epoch_17.pt', map_location=device))

            # Move the model to the correct device
            resnet50_model.to(device)
            resnet50_model.eval()
            outputs = resnet50_model(img_tensor)
            _, predicted_classes = torch.topk(outputs, k=5, dim=1)

        # Convert the predicted class labels to the corresponding dog breeds
        predicted_breeds = [subset[idx] for idx in predicted_classes[0]]

        # Get the probabilities for the top 5 predicted classes
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(outputs)[0][predicted_classes[0]]

        df = pd.DataFrame(list(zip(predicted_breeds, probs.tolist())),
                columns =['Predicted Breeds', 'Probability'], index= [1,2,3,4,5])
        
        # Reverse the order of the DataFrame
        df = df.iloc[::-1]
        fig=px.bar(df,x='Probability',y='Predicted Breeds', orientation='h')
        st.write(fig)
        st.balloons()