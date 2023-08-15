import numpy as np
import pandas as pd
import cv2
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from sklearn.model_selection import train_test_split
from albumentations import Compose, CenterCrop, Normalize
from albumentations.pytorch.transforms import ToTensorV2
import time
from stqdm import stqdm
import streamlit as st
import time
import os
import glob
import re
from multiprocessing import Process
import shutil
from streamlit_extras.dataframe_explorer import dataframe_explorer
from torch.nn.utils import clip_grad_norm

seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

annot_path = '/home/nbui/Public/dog_breed/pages/annotations/Annotation'
image_path = '/home/nbui/Public/dog_breed/pages/images/Images'

valid = []

for element in os.listdir(image_path):
    breed = element.split('-')[1]
    images = len(os.listdir(os.path.join(image_path, element)))
    valid.append((breed, images))

df = pd.DataFrame(valid, columns=['Breeds', 'Number of images'])

# Get all the unique breeds in the dataset
all_breeds = df['Breeds'].unique()

# Modify 'subset' to include all breeds
subset = all_breeds.tolist()

valid = []
for element in os.listdir(image_path):
    if element.split('-')[1] in subset:
        for img_id in os.listdir(os.path.join(image_path, element)):
            path = os.path.join(element, img_id)
            label = element.split('-')[1]
            valid.append((path, label))

df = pd.DataFrame(valid, columns=['Path', 'Label'])

def aug1():
    return Compose([CenterCrop(height=224, width=224, p=1.0), ToTensorV2(Normalize)], p=1)

def aug2():
    return Compose([ToTensorV2(Normalize)], p=1)

train_aug = aug1()
valid_aug = aug2()

def PCAColorAug(image, category='Tensor'):
    if type(image) == torch.Tensor:
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
    
    img_reshaped = image.reshape(-1, 3).astype('float32')
    mean, std = np.mean(img_reshaped, 0), np.std(img_reshaped, 0)
    img_rescaled = (img_reshaped - mean)/std
    cov_matrix = np.cov(img_rescaled, rowvar = False) # Covariant matrix of reshaped image.  Output is 3*3 matrix
    eigen_val, eigen_vec = np.linalg.eig(cov_matrix) # Compute Eigen Values and Eigen Vectors of the covariant matrix. eigen_vec is 3*3 matrix with eigen vectors as column. 
    alphas = np.random.normal(loc = 0, scale = 0.1, size = 3)
    vec1 = alphas*eigen_val
    valid = np.dot(eigen_vec, vec1) # Matrix multiplication
    pca_aug_norm_image = img_rescaled + valid
    pca_aug_image = pca_aug_norm_image*std + mean
    aug_image = np.maximum(np.minimum(pca_aug_image, 255), 0).astype('uint8')
    if category == 'Tensor':
        return torch.from_numpy(aug_image.reshape(3,256,256))
    else:
        return aug_image.reshape(256,256,3)

labels = pd.get_dummies(df['Label'])

# Dataset restructure to fit the Resnet models
class StanfordDogs(Dataset):
    def __init__(self, transform1, transform2, X, Y, objective = 'train'):
        self.X = X
        self.Y = Y
        self.train_transform = transform1
        self.valid_transform = transform2
        self.objective = objective
        
    def __getitem__(self, idx):
        path = self.X['Path'][idx]
        label = self.Y.iloc[idx, :].values
        img = cv2.imread(os.path.join(image_path, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Shortest side of image is scaled to 256 pixels and the other side is scaled so as to maintain aspect ratio
        
        h, w, _ = img.shape
        
        if h <= w:
            aspect_ratio = w/h
            dim = (256, int(256*aspect_ratio))
            img = cv2.resize(img, dim)
        else:
            aspect_ratio = h/w
            dim = (int(256*aspect_ratio), 256)
            img = cv2.resize(img, dim)

           
        img = CenterCrop(height = 256, width = 256, p = 1)(image = img)['image']
        
        if self.objective == 'train':
            random = np.random.uniform(size = 1)
            if random < 0.5:                            # PCA Augmentation carried out only 50 percent of time
                img = PCAColorAug(img, category = 'numpy')
                
            augmented = self.train_transform(image = img)
            img = augmented['image']
            
            return img, label
        
        elif ((self.objective == 'validation') |  (self.objective == 'test')):
            img = cv2.resize(img, (224, 224))
            augmented = self.valid_transform(image = img)
            img = augmented['image']  
            
            return img, label
        
    def __len__(self):
        return len(self.X)

# Split the dataset into train, validation, and test sets
train_X, test_X, train_y, test_y = train_test_split(df, labels, test_size=0.25, random_state=5, stratify=df['Label'])
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.25, random_state=5, stratify=train_X['Label'])

train_X.reset_index(drop=True, inplace=True)
val_X.reset_index(drop=True, inplace=True)
test_X.reset_index(drop=True, inplace=True)

train_y.reset_index(drop=True, inplace=True)
val_y.reset_index(drop=True, inplace=True)
test_y.reset_index(drop=True, inplace=True)

# Define Batch Size
BATCH_SIZE = 64

def loader(data_X, data_Y, batch_size=BATCH_SIZE, obj='train'):
    data = StanfordDogs(train_aug, valid_aug, X=data_X, Y=data_Y, objective=obj)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return loader

train_loader = loader(train_X, train_y, batch_size=BATCH_SIZE)
val_loader = loader(val_X, val_y, batch_size=BATCH_SIZE, obj='validation')

st.title('Training Model')
st.sidebar.title("Controls")

# Perhaps not necessary to store the value
if 'disable_opt' not in st.session_state:
    st.session_state.disable_opt = False

# Create and initialize the chosen ResNet model
emp = st.empty()
resnet_option = emp.radio(
    "Choose your favorite Resnet Model:",
    (" ", 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152'), disabled = st.session_state.disable_opt)

# Check if training has started
training_started = False  # Set this to True once training starts

if resnet_option == "Resnet50":
    resnet_model = resnet50(progress = True, weights=ResNet50_Weights.DEFAULT)
elif resnet_option == "Resnet18":
    resnet_model = resnet18(progress = True, weights=ResNet18_Weights.DEFAULT)
elif resnet_option == "Resnet34":
    resnet_model = resnet34(progress = True, weights=ResNet34_Weights.DEFAULT)
elif resnet_option == "Resnet101":
    resnet_model = resnet101(progress = True, weights=ResNet101_Weights.DEFAULT)
elif resnet_option == "Resnet152":
    resnet_model = resnet152(progress = True, weights=ResNet152_Weights.DEFAULT)
else:
    st.warning('PLease select one of the above Resnet Models', icon="⚠️")

# Display the selected model option
if resnet_option: 
    st.sidebar.markdown(f"Selected ResNet Model: **{resnet_option}**")

if resnet_option != " ":
    # Modify the final fully connected layer for the number of classes
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, len(subset))

    # Convert the model's parameters (weights) to FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model.to(device).float()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing= 0.1)
    optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

    # Create a directory to store the checkpoints
    checkpoint_dir = '/home/nbui/Public/dog_breed/pages/checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    lst = []
    def train_model(model, train_loader, valid_loader, criterion, optimizer, exp_lr_scheduler, epochs):
        since = time.time()
        best_val_accuracy = 0.0

        #check if exists any checkpoint in checkpoints folder
        checkpoint_dir = '/home/nbui/Public/dog_breed/pages/checkpoints/'
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*'))
        if checkpoint_files == []:
            current_epoch = 0
        else:
            def myfunc(e):
                match = re.search(r"(\d+)\.pt$", e)
                if match:
                    current_epoch = int(match.group(1))
                    return current_epoch

                return 0
            checkpoint_files.sort(key= myfunc, reverse= True)

        if checkpoint_files:
            match = re.search(r"(\d+)\.pt$", checkpoint_files[0])
            if match:
                current_epoch = int(match.group(1))
                model.load_state_dict(torch.load(checkpoint_files[0], map_location= device))

            else:
                match1 = re.search(r"(\d+)\.pt$", checkpoint_files[0])
                if match1:
                    current_epoch = int(match1.group(1))
                    model.load_state_dict(torch.load(checkpoint_files[0]))

        epoch_lst = []
        train_loss_lst = []
        val_loss_lst = []
        val_accuracy_lst = []
        for epoch in range(current_epoch + 1, epochs + 1):
            model.train()
            train_loss = 0.0

            progress_bar = stqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for images, labels in progress_bar:
                images, labels = images.to(device).float(), labels.to(device).long()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, torch.argmax(labels, 1))
                loss.backward()
                
                clip_grad_norm(model.parameters(), max_norm=1.0)
                optimizer.step()
                exp_lr_scheduler.step()

                train_loss += loss.item() * images.size(0)

                progress_bar.set_postfix(train_loss=train_loss / (len(progress_bar) * BATCH_SIZE))
            
            model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device).float(), labels.to(device).long()

                    outputs = model(images)
                    loss = criterion(outputs, torch.argmax(labels, 1))
                    val_loss += loss.item() * images.size(0)

                    _, predicted = torch.max(outputs, 1)
                    total_predictions += labels.size(0)
                    correct_predictions += (predicted == torch.argmax(labels, 1)).sum().item()

            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_val_loss = val_loss / len(valid_loader.dataset)
            val_accuracy = 100.0 * correct_predictions / total_predictions
            
            epoch_lst.append(epoch)
            train_loss_lst.append(epoch_train_loss)
            val_loss_lst.append(epoch_val_loss)
            val_accuracy_lst.append(val_accuracy)
            
            # Save the model checkpoint
            checkpoint_filename = os.path.join(checkpoint_dir, f'resnet50_checkpoint_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_filename)
            if os.path.exists(checkpoint_filename):
                st.write(f"Checkpoint {epoch} is successfull stored")

            # Save the best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_checkpoint_filename = os.path.join(checkpoint_dir, 'best_resnet50_checkpoint.pt')
                torch.save(model.state_dict(), best_checkpoint_filename)
        
        time_elapsed = time.time() - since
        st.write('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
        st.subheader("Here's the table summerizing the training process")
        result_df = pd.DataFrame(list(zip(epoch_lst, train_loss_lst, val_loss_lst, val_accuracy_lst)),
               columns =['Epoch', 'Epoch Train Loss', 'Epoch Val Loss', 'Val Accuracy'])
        filtered_df = dataframe_explorer(result_df)
        st.write(filtered_df)

    # Evaluate the model on the test set
    def evaluate_model(model, test_loader, criterion):
        model.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device).float(), labels.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, torch.argmax(labels, 1))
                test_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == torch.argmax(labels, 1)).sum().item()

        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = 100.0 * correct_predictions / total_predictions

        st.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Reduce excessive resources
    EPOCHS = 25
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True 

    # Reduce the learning rate after every 10 epochs until reaches 20
    exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)

    placeholder = st.empty()
    left, right = st.columns(2)
    with left:
        start = placeholder.button('Start', disabled=False, key='1')
    with right:
        stop = right.button("Stop")

    # Create or get the session state using st.session_state
    if "pid" not in st.session_state:
        st.session_state.pid = None

    # Handle the buttons
    if start and st.session_state.pid is None:
        # Start the training process
        placeholder.button('Start', disabled=True, key='2')
        st.sidebar.markdown(f"Let the training of {resnet_option} begin")
        training_started = True
        p = Process(target=train_model(resnet_model, train_loader, train_loader, criterion, optimizer, exp_lr_scheduler, epochs=EPOCHS))
        p.start()
        st.session_state.pid = p.pid
        st.session_state.training_started = True
        st.warning("Please do not choose another model while the training is in process", icon="⚠️")
        # Evaluate the model on the test set
        with st.spinner('Wait for the evaluation process...'):
            time.sleep(20)
        p1 = Process(target=evaluate_model(resnet_model, val_loader, criterion))
        st.snow()

        # path
        path = '/home/nbui/Public/dog_breed/pages/checkpoints'
        # Remove the specified
        # file path
        try:
            shutil.rmtree(path)
            print("% s removed successfully" % path)
        except OSError as error:
            print(error)
            print("File path can not be removed")

    elif stop and st.session_state.pid:
        st.stop()

    # # Disable model selection if training has started
    # if training_started == True:
    #     st.session_state.disable_opt = True
    #     resnet_option = emp.radio(
    #     "Choose your favorite Resnet Model:",
    #     ('Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152'), disabled = st.session_state.disable_opt)
    # else:
    #     st.session_state.resnet_option = resnet_option

    # # Display the model selection dropdown
    # if not training_started:
    #     # Generate a unique key based on the selected model option
    #     key = hash(resnet_option)
    #     st.session_state.resnet_option = resnet_option

