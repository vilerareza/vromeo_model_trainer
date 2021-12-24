from kivy.app import App
from kivy.properties import ObjectProperty

from modeltrainer import ModelTrainer

class LearningApp(App):
    manager = ObjectProperty(None)
    
    def build(self):
        self.manager = ModelTrainer()
        return self.manager
        
LearningApp().run()
