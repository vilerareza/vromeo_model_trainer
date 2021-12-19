from kivy.lang import Builder
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, NumericProperty
from kivy.uix.gridlayout import GridLayout

Builder.load_file('imageviewer.kv')

class ImageViewerBox(BoxLayout):
    labelBox = ObjectProperty(None)
    imageGrid = ObjectProperty(None)
    dataCancelButton = ObjectProperty(None)
    dataConfirmButton = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def print_label(self, label_text):
        self.labelBox.labelBoxLabel.text = label_text

    def button_press_callback(self, widget):
        if widget == self.ids.data_confirm_button:
            widget.source = "images/confirmbutton_down.png"
        elif widget == self.ids.data_cancel_button:
            widget.source = "images/cancelbutton_down.png"

    def button_release_callback(self, widget):
        if widget == self.ids.data_confirm_button:
            widget.source = "images/confirmbutton.png"
        elif widget == self.ids.data_cancel_button:
            widget.source = "images/cancelbutton.png"

class ImageGrid(GridLayout):
    nLive = NumericProperty(0)