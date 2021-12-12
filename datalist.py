import os
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.uix.behaviors import FocusBehavior
from kivy.properties import ObjectProperty 
from kivy.lang import Builder

Builder.load_file('datalist.kv')

from dataitem import DataItem

class DataListBox (BoxLayout):
    dataListLayout = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_item(self, label, fileDir, color):
        imageFiles = os.listdir(fileDir)
        for imageFile in imageFiles:
            filePath = os.path.join(fileDir, imageFile)
            self.dataListLayout.add_widget(DataItem(label, filePath, color))

class DataListLayout (FocusBehavior, CompoundSelectionBehavior, StackLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    


