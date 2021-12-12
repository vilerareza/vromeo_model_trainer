from kivy.graphics import Color, Rectangle
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty

Builder.load_file('datatraining.kv')

class DataTrainingBox(FloatLayout):
   
    nEpoch = NumericProperty(1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize widgets

    def button_press_callback(self, button):
        if button == self.ids.training_start_button:
            button.source = "images/starttraining_down.png"
        elif button == self.ids.save_model_button:
            button.source = "images/savemodel_down.png"
        elif button == self.ids.epoch_up_button:
            button.source = "images/epochup_down.png"
        elif button == self.ids.epoch_down_button:
            button.source = "images/epochdown_down.png"

    def button_release_callback(self, button):
        if button == self.ids.training_start_button:
            button.source = "images/starttraining.png"
        elif button == self.ids.save_model_button:
            button.source = "images/savemodel.png"
        elif button == self.ids.epoch_up_button:
            button.source = "images/epochup.png"
            self.change_epoch(button)
        elif button == self.ids.epoch_down_button:
            button.source = "images/epochdown.png"
            self.change_epoch(button)

    def change_epoch(self, button):
        if button == self.ids.epoch_up_button:
            if self.nEpoch < 20:
                self.nEpoch += 1
        elif button == self.ids.epoch_down_button:
            if self.nEpoch > 1:
                self.nEpoch -= 1

    def display_accuracy(self, accuracy = ''):
        self.ids.training_result_label.text = f'Training Accuracy Result (%): {accuracy}'
        self.ids.training_result_label.opacity = 1
    
    def hide_accuracy(self):
        self.ids.training_result_label.text = ''
        self.ids.training_result_label.opacity = 0
