import os
import random
import uuid
import threading
import pickle
import numpy as np

from tkinter import Tk, filedialog
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model 
from tensorflow.keras import optimizers
from cv2 import imwrite, rectangle
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import NumericProperty, ObjectProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar 
from kivy.clock import Clock, mainthread
from kivy.graphics import Color, Rectangle

from dataentry import DataEntryBox
from datalist import DataListBox
from datatraining import DataTrainingBox
from imageprocessor import ImageProcessor
from imageviewer import ImageViewerBox
from logobar import LogoBar
from mywidgets import ImageButton, ButtonBinded

Builder.load_file('modeltrainer.kv')

class ModelTrainer(BoxLayout):
    middleContainerBox = ObjectProperty(None)
    leftBox = ObjectProperty(None)
    imageViewerBox = ObjectProperty(None)
    dataEntryBox = ObjectProperty(None)
    dataListBox = ObjectProperty(None)
    dataTrainingBox = ObjectProperty(None)
    dataDeletedList = ListProperty([])

    selectedPath = ''
    newLabel = ''
    
    model = None    # Recognition model
    imageProcessor = None   # Detector
    nEpoch = NumericProperty(1)
    
    dataset = {}
    datasetImages = []
    datasetLabels = []
    nClasses = 0

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        # Initialize layout
        self.logoBar = LogoBar(color=(0,0,0), size_hint = (1,None), height = 40)
        self.middleContainerBox = BoxLayout (orientation = 'horizontal')
        self.leftBox = BoxLayout(orientation = "vertical",size_hint = (0.9,1))
        self.dataEntryBox = DataEntryBox()
        self.dataListBox = DataListBox()
        self.imageViewerBox = ImageViewerBox ()
        self.dataTrainingBox = DataTrainingBox()
        
        # Adding to left box
        self.leftBox.add_widget(self.dataEntryBox)
        self.leftBox.add_widget(self.imageViewerBox)

        # Adding to container box
        self.middleContainerBox.add_widget(self.leftBox)
        self.middleContainerBox.add_widget(self.dataListBox)

        # Progress bar
        self.progressBar = ProgressBar()
        self.progressPop = Popup(content = self.progressBar,
                                 size_hint=(0.5, 0.2),
                                 auto_dismiss = False)
        self.progress_value = 0
        self.thread_flag = False

        # Show logo bar
        self.add_widget(self.logoBar)
        # Show container box
        self.add_widget(self.middleContainerBox)
        # Show data training box
        self.add_widget(self.dataTrainingBox)

        #delete button
        self.deleteButton = ButtonBinded(background_normal = "images/cancelbutton.png",
                                        size_hint = (None, None),
                                        size = (20, 20))
       
    def progressUp(self, target_function):
        print(target_function)
        self.progressPop.open()
        self.thread_flag = True
        self.ev = threading.Event()
        self.thread = threading.Thread(target = target_function)
        self.thread.start()
    
    def dismiss(self, *args):
        self.ev.clear()
        self.thread_flag = False
        self.progressPop.dismiss()
    
    def clock(self, *args):
        #Stops the clock when the progress bar value reaches its maximum
        if self.progress_value >= 100:
            Clock.unschedule(self.clock)
        self.progressBar.value = self.progress_value

    
    def review_data(self):
        # Initialize progress bar value
        self.progress_value = 0
        Clock.schedule_interval(self.clock, 1 / 60)
        self.progressPop.title = 'Recognize your images'
        if self.thread_flag:
            # Get data directory and path
            self.selectedPath, self.newLabel = self.get_data_entry()

            # Create image processor object if none
            if not self.imageProcessor:
                self.imageProcessor = self.create_image_processor()

            # Clear previous data
            self.clear_preview_images('images/temp/preview/', self.imageViewerBox.imageGrid)

            # Process the data for review
            if (os.path.isdir(self.selectedPath) and self.newLabel!=''):
                # Directory and label valid
                imageFiles = os.listdir(self.selectedPath)
                self.progress_step = round(90 / (len(imageFiles * 2)))
                self.imageViewerBox.print_label(self.newLabel)
                self.progress_value += 5
                for imageFile in imageFiles:
                    filePath = os.path.join(self.selectedPath, imageFile)
                    img, detectionBox = self.imageProcessor.detect_face(filePath)
                    self.progress_value += self.progress_step
                    self.create_preview_image(img, box = detectionBox)
                    self.progress_value += self.progress_step    
                
                self.draw_image_to_grid(self.imageViewerBox.imageGrid)
                self.progress_value += 5
            else:
                print ('Unable to process. Invalid path or empty label')

        self.dismiss()

    def get_data_entry(self):
        # Get selected directory
        selectedPath = self.dataEntryBox.selectedPath
        # Getting label from input text
        label = self.dataEntryBox.dataLabelText.text
        print(f'Selected Path: {selectedPath}, Label: {label}')
        return selectedPath, label

    def create_image_processor(self):
        return ImageProcessor()

    def select_image(self, widget):
        print(widget)
        if widget.opacity == 1:
            widget.opacity = 0.3
            self.dataDeletedList.append((widget.source, widget))
            print(self.dataDeletedList)
        else:
            widget.opacity = 1
            self.dataDeletedList.remove((widget.source, widget))
            print(self.dataDeletedList)
        
        #enable cancel button
        if len(self.dataDeletedList) !=0:
            self.imageViewerBox.dataCancelButton.disabled = False
        else:
            self.imageViewerBox.dataCancelButton.disabled = True

    def delete_image(self, widget):
        gridLayout = self.imageViewerBox.imageGrid
        for image in self.dataDeletedList:
            os.remove(image[0])
            gridLayout.remove_widget(image[1])
            gridLayout.nLive -= 1
            print(gridLayout.nLive)

        nLiveMin = (gridLayout.rows-1)**2 + (gridLayout.rows-1)
        if gridLayout.nLive <= nLiveMin:
            gridLayout.rows -=1
        
        self.dataDeletedList.clear()

        
    @mainthread
    def draw_image_to_grid(self, gridLayout):
        imagePath = 'images/temp/preview/'
        imageFiles = os.listdir(imagePath)
        gridLayout.nLive = 0
        if len(imageFiles)>0:
            for imageFile in imageFiles:
                # Arrange self.imageViewerBox.imageGrid cols and rows
                gridLayout.nLive +=1
                nLiveMax = gridLayout.rows**2 + gridLayout.rows
                # Add row to grid item if already reach maximum item
                if gridLayout.nLive > nLiveMax:
                    gridLayout.rows +=1
                # Add the image
                gridLayout.add_widget(ImageButton(source = os.path.join(imagePath, imageFile), 
                                                  on_press = self.select_image))
        
        self.imageViewerBox.dataConfirmButton.disabled = False

    def create_preview_image(self, img, box):
        # Generate random uuid for file name
        uuidName = uuid.uuid4()
        previewImagePath = (f'images/temp/preview/{uuidName}.png')
        if box:
            xb, yb, widthb, heightb = box
            rectangle(img, (xb, yb), (xb+widthb, yb+heightb), color = (232,164,0), thickness = 3)
        imwrite(previewImagePath, img)
        return previewImagePath, img

    def clear_preview_images(self, imagesLocation, gridLayout = None):
        images = os.listdir(imagesLocation)
        for image in images:
            os.remove(os.path.join(imagesLocation, image))
        if gridLayout:
            gridLayout.clear_widgets()

    def add_data(self, dataset, label, faceList):
        dataset[label] = faceList

    def save_data(self, widget):
        root = Tk()
        root.withdraw()
        filename = filedialog.asksaveasfilename(defaultextension='.pckl', filetypes= [('pickle files','*.pckl')])
        root.destroy()
        if filename:
            with open(filename, 'wb') as file:
                pickle.dump(self.dataset, file)
            self.dataListBox.saveFile.disabled = True
            self.clear_preview_images('images/temp/dataset/', self.dataListBox.dataListLayout)
        else:
            print("No data saved")
    
    def load_data(self, widget):
        self.clear_preview_images('images/temp/dataset/', self.dataListBox.dataListLayout)
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(filetypes= [("pickle files","*.pckl")])
        root.destroy()
        if filename:
            file = open(filename, "rb")
            loaded_dataset = pickle.load(file)
            try:
                for label in loaded_dataset.keys():
                    self.add_to_list(loaded_dataset, label)
                    print('data loaded')
            except:
                print("Sorry, we couldnt load your file")
            
        else:
            print("No data loaded")
    
    def refresh(self):
        self.clear_preview_images('images/temp/preview/', self.imageViewerBox.imageGrid)
        self.imageViewerBox.print_label("")
        self.imageViewerBox.dataCancelButton.disabled = True
        self.imageViewerBox.dataConfirmButton.disabled = True
        self.dataEntryBox.dataLabelText.text = ""
        self.dataEntryBox.dataLocationText.text = ""
        
    @mainthread
    def add_to_list(self, dataset, label):
        # Prepare random color for databox in the list
        dataColor = (random.random(), random.random(), random.random())
        faceList = dataset[label]
        # Create file for every image data in faceList
        saveTempPath = 'images/temp/dataset/'
        self.clear_preview_images(saveTempPath)
        for img in faceList:
            uuidName = uuid.uuid4()
            writePath = (f'{saveTempPath}{uuidName}.png')
            imwrite(writePath, img)
        # draw the new data to the list 
        self.dataListBox.add_item(label, saveTempPath, dataColor)
        self.dataListBox.saveFile.disabled = False
        self.dataTrainingBox.trainingButton.disabled = False
        self.refresh()

    def add_to_dataset(self):
        self.progress_value = 0
        Clock.schedule_interval(self.clock, 1 / 60)
        self.progressPop.title = 'Adding your images to the database'
        path = 'images/temp/preview'
        if self.thread_flag:
            imageFiles = os.listdir(path)
            self.progress_step = round(90 / len(imageFiles))
            faceList=[]
            for imageFile in imageFiles:
                filePath = os.path.join(path, imageFile)
                print(filePath)
                face = self.imageProcessor.extract_face(filePath)
                if face is not None:
                    faceList.append(face)
                    self.progress_value += self.progress_step

            self.add_data(dataset = self.dataset, label = self.newLabel, faceList = faceList)
            self.progress_value += 5
            self.add_to_list(dataset = self.dataset, label = self.newLabel)
            self.progress_value += 5
        self.dismiss()

    def start_model_training(self):
        # get training parameter: model, epoch
        if not self.model:
            self.nClasses = len(self.dataset)
            self.model = self.create_model(self.nClasses)
            print (f'nClasses: {self.nClasses}')
        if self.model:
            # Get epoch
            self.nEpoch = self.get_epoch()
            # Configure for training
            self.model.compile(optimizer= optimizers.Adam(learning_rate = 1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            # Get training data
            self.datasetImages, self.datasetLabels = self.get_training_data()
            print (str(len(self.datasetImages))+', '+str(type(self.datasetImages)))
            print (str(len(self.datasetLabels))+', '+str(type(self.datasetLabels)))
            # Start the training
            # import pickle
            # pickle.dump(self.datasetImages,open("datasetimages.p", "wb"))
            # pickle.dump(self.datasetLabels,open("datasetlabels.p", "wb"))
            # print('pickle done')
            history = self.model.fit(self.datasetImages, self.datasetLabels, epochs = self.nEpoch, batch_size = len(self.datasetImages))   #Batch Size?
            accuracy = int((history.history['categorical_accuracy'][-1])*100)
            self.dataTrainingBox.display_accuracy(str(accuracy))
            self.dataTrainingBox.saveModelButton.disabled = False
            #print (str(accuracy))

    def create_model (self, nClasses):
        from vggface import VGGFace
        model = VGGFace(nClasses).model
        return model
    
    def get_epoch(self):
        return self.dataTrainingBox.nEpoch
    
    def get_training_data(self):
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

    def save_model_to_file(self):
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
