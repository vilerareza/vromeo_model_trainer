from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image

class LogoBar (FloatLayout):
    def __init__(self, color, **kwargs):
        super().__init__(**kwargs)
        # Adding image, original size is 800x80
        self.add_widget(Image(source = "images/logo.png", mipmap = True, pos_hint = {'left': 1, 'center_y': 0.5}, size_hint = (None, None), size = (300, 30)))
        
        with self.canvas.before:
            Color(color[0],color[1],color[2])
            self.rect = Rectangle (pos=self.pos, size = self.size)
            self.bind (pos = self.update_rect, size = self.update_rect)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size