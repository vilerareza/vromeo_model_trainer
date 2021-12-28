import random
from kivy.graphics import Color, Rectangle
from kivy.lang import Builder
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors import ButtonBehavior

Builder.load_file('dataitem.kv')

class DataItem(ButtonBehavior ,BoxLayout):
    selected = BooleanProperty (False)
    dataLabel = ObjectProperty(None)
    dataImage = ObjectProperty(None)
    
    def __init__(self, label, filePath, color, **kwargs):
        super().__init__(**kwargs)
        self.dataLabel.text = label
        self.dataImage.source = filePath

        with self.canvas.before:
            Color(color[0],color[1],color[2])
            self.rect = Rectangle (pos=self.pos, size = self.size)
            self.bind (pos = self.update_rect, size = self.update_rect)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
