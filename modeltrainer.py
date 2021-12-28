import os
import pickle
import random
import threading
import uuid
from tkinter import Tk, filedialog
from typing import List, final

import numpy as np
from cv2 import imread, imwrite, rectangle
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.properties import ListProperty, NumericProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from sklearn import preprocessing

from mywidgets import ImageButton
from progressbox import ProgressBox

Builder.load_file('modeltrainer.kv')

class ModelTrainer(BoxLayout):
    middleContainerBox = ObjectProperty(None)
    leftBox = ObjectProperty(None)
    imageViewerBox = ObjectProperty(None)
    dataEntryBox = ObjectProperty(None)
    dataListBox = ObjectProperty(None)
    dataTrainingBox = ObjectProperty(None)

    progressBox = ObjectProperty(None)
    progressPop = ObjectProperty(None)
    progress_value = 0

    selectedPath = ''
    newLabel = ''
    previewLocation = 'images/temp/preview/'
    preview_face_folder = 'images/temp/preview_face/'
    dataset_buffer_folder = 'images/temp/dataset_buffer/'

    model = None    # Recognition model
    imageProcessor = None   # Detector
    nEpoch = NumericProperty(1)
    
    dataset = {}
    datasetImages = []
    datasetLabels = []
    nClasses = 0
    dataToDelete = {}
    imageToDelete = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Progress box
        self.progressBox =  ProgressBox()
        self.progressPop = Popup(content = self.progressBox, size_hint=(0.5, 0.2), auto_dismiss = False)
        # Bindings
        self.bind(imageToDelete = self.update_preview_clear_button)

    def progressUp(self, target_function):
        # For review button: target_function = self.review_data()
        self.progressPop.open()
        thread = threading.Thread(target = target_function)
        thread.start()
    
    def dismiss(self, *args):
        self.progressPop.dismiss()
    
    def clock(self, *args):
        #Stops the clock when the progress bar value reaches its maximum
        if self.progress_value >= 100:
            Clock.unschedule(self.clock)
        self.progressBox.progressBar.value = self.progress_value

    def get_data_entry(self):
        # Get selected directory
        selectedPath = self.dataEntryBox.selectedPath
        # Getting label from input text
        label = self.dataEntryBox.dataLabelText.text
        print(f'Selected Path: {selectedPath}, Label: {label}')
        return selectedPath, label

    def create_image_processor(self):
        from imageprocessor import ImageProcessor
        return ImageProcessor()

    def create_preview_image(self, img, box):
        # Generate random uuid for file name
        uuidName = uuid.uuid4()
        previewImagePath = (f'{self.previewLocation}{uuidName}.png')
        if box:
            xb, yb, widthb, heightb = box
            rectangle(img, (xb, yb), (xb+widthb, yb+heightb), color = (232,164,0), thickness = 3)
        imwrite(previewImagePath, img)
        return uuidName, img

    @mainthread
    def draw_image_to_grid(self, imagePath, gridLayout):    # REVIEW OK
        imageFiles = os.listdir(imagePath)
        if len(imageFiles)>0:
            for imageFile in imageFiles:
                # Arrange self.imageViewerBox.imageGrid cols and rows
                gridLayout.nLive +=1
                nLiveMax = gridLayout.rows**2 + gridLayout.rows
                # Add row to grid item if already reach maximum item
                if gridLayout.nLive > nLiveMax:
                    gridLayout.rows +=1
                # Add the image and bound to method when selected
                gridLayout.add_widget(ImageButton(source = os.path.join(imagePath, imageFile), fileName = imageFile,
                                                  on_press = self.select_image))

    def clear_images(self, imagesLocation = '', gridLayout = None):  #REVIEW OK
        # Removing image files in temp preview directory 
        if imagesLocation != '':
            images = os.listdir(imagesLocation)
            for image in images:
                os.remove(os.path.join(imagesLocation, image))
        # Clearing displayed images in image viewer grid layout
        if gridLayout:
            gridLayout.clear_widgets()
            gridLayout.nLive = 0

    def review_data(self):
        # Initialize progress bar value
        self.progress_value = 0
        Clock.schedule_interval(self.clock, 1 / 10)
        self.progressPop.title = 'Loading images...'
        # Get data directory and path
        self.selectedPath, self.newLabel = self.get_data_entry()
        # Create image processor object if none
        if not self.imageProcessor:
            self.imageProcessor = self.create_image_processor()
        # Clear previous data in the preview directory and clear delete list anyway
        self.clear_images(self.previewLocation, self.imageViewerBox.imageGrid)
        self.clear_images(self.preview_face_folder, self.imageViewerBox.imageGrid)
        self.imageToDelete.clear()
        self.update_preview_clear_button()

        # Process the data for review
        if (os.path.isdir(self.selectedPath) and self.newLabel!=''):
            # Directory and label valid
            self.imageViewerBox.print_label(self.newLabel)  # Printing label to image viewer box
            imageFiles = os.listdir(self.selectedPath)
            progress_step = round(90 / (len(imageFiles * 2)))   #Progress step calculation
            self.progress_value += 5    # Increase progress by 5% after printing label and lisitng files
            # Detect face in every image
            for imageFile in imageFiles:
                filePath = os.path.join(self.selectedPath, imageFile)
                img, detectionBox = self.imageProcessor.detect_face(filePath)
                self.progress_value += progress_step    # Increase progress after detection
                fileName, img = self.create_preview_image(img, box = detectionBox)
                if detectionBox:
                    face = self.imageProcessor.extract_face(filePath)
                    if np.any(face):
                        writePath = (f'{self.preview_face_folder}{fileName}.png')
                        imwrite(writePath, face)
                self.progress_value += progress_step    # Increase progress after creating preview image
            self.draw_image_to_grid(imagePath = self.previewLocation, gridLayout = self.imageViewerBox.imageGrid)
            self.progress_value += 5    # Increase progress by 5% after drawing to image viewer    
        else:
            print ('Unable to process. Invalid path or empty label')
            
        self.dismiss()

    def select_image(self, image):  # REVIEW OK
        if not image.selected:
            # Selected
            image.selected = True
            image.opacity = 0.3
            self.imageToDelete.append((image.source, image))
        else:
            # Not selected
            image.selected = False
            image.opacity = 1
            self.imageToDelete.remove((image.source, image))

    def delete_selected_image(self, widget): # REVIEW OK
        # Delete selected images in image viewer box
        if (len(self.imageToDelete) > 0):
            gridLayout = self.imageViewerBox.imageGrid
            for image in self.imageToDelete:
                # Remove the preview image file
                os.remove(image[0])
                 # Remove the image in preview_face_location (if any)
                try:
                    os.remove(f'{self.preview_face_folder}{image[1].fileName}')
                except Exception as e:
                    print (f'{e} : The file may not be exist but is is okay.')
                # Remove the image widget from the grid layout and remove the image file from the preview image location
                gridLayout.remove_widget(image[1])
            # Rearrange grid rows
            gridLayout.nLive -= 1
            nLiveMin = (gridLayout.rows-1)**2 + (gridLayout.rows-1)
            if (gridLayout.nLive <= nLiveMin):
                if gridLayout.rows > 1:
                    gridLayout.rows -=1
            # Clear delete list
            self.imageToDelete.clear()
            self.update_preview_clear_button()

    def update_preview_clear_button(self, *args):   #REVIEW OK
        if (len(self.imageToDelete)) > 0:
            self.imageViewerBox.dataCancelButton.disabled = False
        else:
            self.imageViewerBox.dataCancelButton.disabled = True

    def add_to_dataset(self, dataset, label, faceList):   # REVIEW OK
        # Adding new data and label to dataset
        dataset[label] = faceList
        # Enable the training
        self.check_training_ready(self.dataset, self.dataTrainingBox)

    def add_to_list(self, label, image_location):  #REVIEW OK
        # Prepare random color for databox in the list
        dataColor = (random.random(), random.random(), random.random())
        # draw the new data to the list 
        self.dataListBox.add_item(label = label, fileDir = image_location, color = dataColor)
        # Refresh some widgets
        self.refresh()

    def add_data(self):   # REVIEW OK
        self.progress_value = 0     # Initialize progress bar value
        Clock.schedule_interval(self.clock, 1 / 5)
        self.progressPop.title = 'Adding images to dataset...'
        imageFiles = os.listdir(self.preview_face_folder)
        progress_step = round(90 / len(imageFiles))     # Progress step calculation
        faceList=[]
        for imageFile in imageFiles:
            filePath = os.path.join(self.preview_face_folder, imageFile)
            face = imread(filePath)
            faceList.append(face)
            self.progress_value += progress_step
        # Adding face data and label to dataset
        self.add_to_dataset(dataset = self.dataset, label = self.newLabel, faceList = faceList)
        self.progress_value += 5    # Increase progress by 5% after adding to dataset
        self.add_to_list(label = self.newLabel, image_location = self.preview_face_folder)
        self.clear_images(self.previewLocation, self.imageViewerBox.imageGrid)
        self.progress_value += 5    # Increase progress by 5% after adding to the list for display
        self.dismiss()

    def refresh(self):  # REVIEW OK
        self.clear_images(self.previewLocation, self.imageViewerBox.imageGrid)
        self.imageViewerBox.print_label("")
        self.dataEntryBox.dataLabelText.text = ""
        self.dataEntryBox.dataLocationText.text = ""

    def select_data(self, widget):      # REVIEW OK
        # Widget is dataItem object
        dataItemList = []
        for child in self.dataListBox.dataListLayout.children:
            if child.dataLabel.text == widget.dataLabel.text:
                theKey = widget.dataLabel.text
                if not child.selected:  # Select
                    child.selected = True
                    dataItemList.append(child)
                    self.dataToDelete[theKey] = dataItemList
                else:   # Unselected
                    child.selected = False
                    self.dataToDelete.pop(theKey, None)
        # Change the dataListBox deleteFile property if there is dataItem selected
        if len(self.dataToDelete) != 0:
            self.dataListBox.deleteFile = True
        else:
            self.dataListBox.deleteFile = False
    
    def delete_data(self, widget):  # REVIEW OK
        # Delete selected data in the dataset and data list layout
        if len(self.dataToDelete) > 0:
            for key, values in self.dataToDelete.items():
                # Remove the data item from the data list layout
                for value in values:
                    self.dataListBox.dataListLayout.remove_widget(value)
                # Remove the data from dataset
                self.dataset.pop(key, None)
            # Clearing the deletion list
            self.dataToDelete.clear()
            self.dataListBox.deleteFile = False
            # Check if the training can be enabled
            self.check_training_ready(self.dataset, self.dataTrainingBox)
        else:
            print ('Nothing to delete. dataToDelete list is empty')

    def save_data(self, widget):    # REVIEW OK
        if (len(self.dataset) > 0):
            root = Tk()
            root.withdraw()
            filename = filedialog.asksaveasfilename(defaultextension='.pckl', filetypes= [('pickle files','*.pckl')])
            root.destroy()
            if filename:
                with open(filename, 'wb') as file:
                    pickle.dump(self.dataset, file)
                #self.clear_images(self.preview_face_folder, self.dataListBox.dataListLayout) Should be removed
            else:
                print("No data saved")
        else:
            print ('Nothing to save. Dataset is empty')
    
    def load_data(self, widget):    # REVIEW OK
        # Backup current database first
        dataset_backup = self.dataset.copy()
        # File selection
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(filetypes= [("pickle files","*.pckl")])
        root.destroy()
        if filename:
            # Clearing existing dataset
            self.dataset.clear()
            # Clearing layout
            self.clear_images(gridLayout = self.dataListBox.dataListLayout)
            # Processing selected file
            file = open(filename, "rb")
            try:
                loaded_dataset = pickle.load(file)
                self.dataset = loaded_dataset.copy()    # Copy the dataset from file
                for label in self.dataset.keys():
                    self.clear_images(self.dataset_buffer_folder)
                    self.display_dataset(dataset = self.dataset, label = label)
            except Exception as e:
                print(f"{e}: Failed loading dataset. Possible cause: wrong dataset file selected")
                self.dataset = dataset_backup.copy()   # Restore database to previous state
                for label in self.dataset.keys():
                    self.clear_images(self.dataset_buffer_folder)
                    self.display_dataset(dataset = self.dataset, label = label)
            finally:
                dataset_backup = None
                # Check if the training can be enabled
                self.check_training_ready(self.dataset, self.dataTrainingBox)
        else:
            print("Selection canceled. No data loaded...")
            dataset_backup = None
    
    def display_dataset(self, dataset, label):      # REVIEW OK
        for img in dataset[label]:
            uuidName = uuid.uuid4()
            writePath = (f'{self.dataset_buffer_folder}{uuidName}.png')
            imwrite(writePath, img)
        dataColor = (random.random(), random.random(), random.random())     # Prepare random color for databox in the list
        self.dataListBox.add_item(label = label, fileDir = self.dataset_buffer_folder, color = dataColor)

    def check_training_ready(self, dataset, widget):    # REVIEW OK
        print (f'len {len(dataset)}')
        if len(dataset) > 0:
            widget.trainingEnabled = True
        else:
            widget.trainingEnabled = False

    def start_model_training(self):
        from tensorflow.keras import optimizers
        self.progress_value = 0     # Initialize progress value
        Clock.schedule_interval(self.clock, 1 / 3)
        self.progressPop.title = 'AI model training in progress...'
        # Create model and get eopch
        self.progressBox.progressLabel.text = '1 : Preparing AI model'
        if not self.model:
            try:
                self.nClasses = len(self.dataset)
                self.model = self.create_model(self.nClasses)
                print (f'nClasses: {self.nClasses}')
            except Exception as e:
                print (f'{e}: Failed creating AI model')
        if self.model:
            # Get epoch
            self.nEpoch = self.get_epoch()
            # Configure for training
            self.model.compile(optimizer= optimizers.Adam(learning_rate = 1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            self.progress_value += 5    # Increasing progress: Model is ready
            # Get training data
            self.progressBox.progressLabel.text = '2 : Getting training data'
            self.datasetImages, self.datasetLabels = self.get_training_data()
            self.progress_value += 5    # Increasing progress: Training data is ready
            print (str(len(self.datasetImages))+', '+str(type(self.datasetImages)))
            print (str(len(self.datasetLabels))+', '+str(type(self.datasetLabels)))
            progress_step = round(85 / self.nEpoch)
            print (f'progress value: {self.progress_value}')
            for epoch in range(self.nEpoch):
                self.label_step.text = f'Training your data {epoch + 1} / {self.nEpoch}'
                history_model = self.model.fit(self.datasetImages, self.datasetLabels, epochs = 1, batch_size = 5)   #Batch Size?
                self.progress_value += progress_step
                print (f'progress value: {self.progress_value}')
            accuracy = int((history_model.history['categorical_accuracy'][-1])*100)
            self.progress_value += 5
            print (f'progress value: {self.progress_value}')
            self.label_step.text = 'Count your training accuracy'
            self.show_accuracy(accuracy)
                
        self.dismiss()

    def create_model (self, nClasses):
        from vggface import VGGFace
        model = VGGFace(nClasses).model
        return model
    
    def get_epoch(self):
        return self.dataTrainingBox.nEpoch
    
    def get_training_data(self):
        from tensorflow.keras.utils import to_categorical
        datasetLabels = []
        datasetImages = []
        for label in self.dataset:
            # Do for every label
            for faceImage in self.dataset[label]:
                datasetLabels.append(label)
                # Get images for a label
                # Normalize face image first
                faceImage = faceImage/255
                datasetImages.append(faceImage)
        # Return collected images and labels        
        datasetImages = np.array(datasetImages)
        # Encode the label
        preprocess_label = preprocessing.LabelEncoder()
        datasetLabels = preprocess_label.fit_transform(datasetLabels)
        datasetLabels = to_categorical(datasetLabels, num_classes = self.nClasses)
        return datasetImages, datasetLabels

    @mainthread
    def show_accuracy(self, accuracy):
        self.dataTrainingBox.display_accuracy(str(accuracy))
        self.dataTrainingBox.saveModelButton.disabled = False

    def save_model_to_file(self):
        from tensorflow.keras.models import save_model 
        try:
            if self.model:
                root = Tk()
                root.withdraw()
                fileName = filedialog.asksaveasfilename(defaultextension='.h5', initialfile = 'vromeo_ai_model', filetypes= [("h5 files","*.h5")])
                if fileName:
                    #self.model.save(fileName)
                    save_model (self.model, fileName, include_optimizer = False)
                root.destroy()
                print (f'Model saved as {fileName}')
            else:
                print ('Nothing to save, model not exist')
        except Exception as e:
            print (e)
        
        self.dataTrainingBox.saveModelButton.disabled = True
