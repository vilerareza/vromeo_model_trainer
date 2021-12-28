from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.properties import ObjectProperty, NumericProperty
from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.uix.behaviors import FocusBehavior

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


class ImageGrid(GridLayout): # FocusBehavior, CompoundSelectionBehavior):
    nLive = NumericProperty(0)
    # selectedImage = ObjectProperty(None)
    # touch_multiselect = True

    # def add_widget(self, widget):
    #     super().add_widget(widget)
    #     widget.bind(on_touch_down = self.widget_touch_down, on_touch_up = self.widget_touch_up)
    
    # def widget_touch_down(self, widget, touch):
    #     if widget.collide_point(*touch.pos):
    #         self.select_with_touch(widget, touch)
    
    # def widget_touch_up(self, widget, touch):
    #     if self.collide_point(*touch.pos) and (not (widget.collide_point(*touch.pos) or self.touch_multiselect)):
    #         self.deselect_node(widget)
    
    # def select_node(self, node):
    #     node.opacity = 0.3
    #     return super().select_node(node)
        
    # def deselect_node(self, node):
    #     super().deselect_node(node)
    #     node.opacity = 1
    #     return super().clear_selection()