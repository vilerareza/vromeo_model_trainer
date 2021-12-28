from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.lang import Builder

from tkinter import Tk, filedialog

Builder.load_file("dataentry.kv")

class DataEntryBox(FloatLayout):
    titleLabel = ObjectProperty(None)
    dataLocationText = ObjectProperty(None)
    dataLocationButton = ObjectProperty(None)
    dataLabelText = ObjectProperty(None)
    dataReviewButton = ObjectProperty(None)
    selectedPath = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def button_press_callback(self, widget):
        if widget == self.dataLocationButton:
            widget.source = "images/selectfile_down.png"
        elif widget == self.dataReviewButton:
            widget.source = "images/reviewbutton_down.png"
    
    def button_release_callback(self, widget):
        if widget == self.dataLocationButton:
            widget.source = "images/selectfile.png"
        elif widget == self.dataReviewButton:
            widget.source = "images/reviewbutton.png"
    
    def show_load_dialog(self, widget):
        root = Tk()
        root.withdraw()
        dirname = filedialog.askdirectory()
        root.destroy()
        if dirname:
            self.load_dir(dirname)

    def load_dir(self, dirname):
        # get selection
        self.selectedPath = dirname
        self.dataLocationText.text = self.selectedPath
        
