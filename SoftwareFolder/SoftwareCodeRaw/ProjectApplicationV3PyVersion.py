# Kivy library & Dependencies
import kivy
kivy.require('2.1.0')

from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.write()

from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock

# Graphic related additions.
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import AsyncImage

from kivy.graphics import Color, Rectangle

#Basics
import os
import io
import re
import string
from tqdm import tqdm
import numpy as np

#torch libraries
import torch

#Time related libraries
import datetime
from timeit import default_timer as timer

#General libraries
import collections
import math
import random

#Transformers Library. Taken from "Hugging Face".
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer, BertForMaskedLM
from tokenizers import BertWordPieceTokenizer


class RootWidget(BoxLayout):
    # Box Layout class which is used as an empty Layout by the "Root"* as the bottom layer (base).
    pass

class CustomLayout(FloatLayout): 
    # Custom Layout class which is used for adding widgets. Similar to putting up sticky notes on a Bulletin Board.

    def __init__(self, **kwargs):   
        super(CustomLayout, self).__init__(**kwargs)

        with self.canvas.before: # Defines the size and position for the Layout.
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self._update_rect, pos=self._update_rect) 
        # Binding the update function to the layout, Allowing to update both the size and position later on if needed.

    def _update_rect(self, instance, value): # Updates the position and size of the Layout to a different value.
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class FindingWordo(App):

    def build(self): # Constructor function which configures and defines the application's GUI, functionalities, etc.
        AppFolderPath = os.path.dirname(__file__)
        self.icon = AppFolderPath + "\ApplicationResources\IconTemp.png" # Application Icon.
        TextInputThing = ''

        Tokenizer = AutoTokenizer.from_pretrained("Seraphiive/bert-personalized-PreAlpha-uncased") 
        Model = BertForMaskedLM.from_pretrained("Seraphiive/bert-personalized-PreAlpha-uncased")
        # Loading the Tokenizer & Model from the HuggingFace library.

        self.ExampleSentence = ''
        self.MissingWord = ''
        self.RecentlyCompleted = ''


        def PrepareSentence(TextToMod): 
            NewText = TextToMod.replace("___", "[MASK]")
            # Replacing the empty space in the sentence with "[MASK]" before processing the sentence,
            # Since the only reason for using "___" when giving a sentence is due to UX considerations.

            return NewText

        def FindMissingWord(SentenceText):
            self.ExampleSentence = PrepareSentence(SentenceText) 
            # Takes a user-given sentence and passes it to the PrepareSentence function before using it for the model.

            inputs = Tokenizer(self.ExampleSentence, return_tensors="pt") # Tokenizing the sentence after modifications.

            with torch.no_grad():
                logits = Model(**inputs).logits
            # Using the Model's inputs to get the "Raw Prediction Vector" which will be used for predicting the token_id.
            # The same token_id which will, (After decoding) provide the missing word.

            mask_token_index = (inputs.input_ids == Tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0] 
            # Retrieving the index of the [MASK] token.


            predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            # Retrieving the token id that represents the “Missing Word”.

            self.MissingWord = Tokenizer.decode(predicted_token_id) 
            # Decoding the predicted token id to text form to know the word.

            self.RecentlyCompleted = self.ExampleSentence.replace("[MASK]", self.MissingWord)
            # Replacing the empty space in a sentence with the predicted word for GUI purposes.


            return True 
            # Only used for ending the function. (The values from the function are being saved to objects within the class).

        root = RootWidget()
        c1 = CustomLayout()
        root.add_widget(c1)
        
        c1.add_widget(
            AsyncImage(
                source=AppFolderPath + "\ApplicationResources\BackgroundTemp.png",
                size_hint=(800, 600),
                pos_hint={'center_x': .5, 'center_y': .5}))

        

        EnterASen = Label(
            text='Enter a sentence;',
            size_hint=(.5, .8),
            pos_hint={'center_x': .32, 'center_y': .5})
        c1.add_widget(EnterASen)

        SentenceIn = TextInput(
            text = '',
            multiline=False,
            size_hint=(.4, .12),
            pos_hint={'center_x': .62, 'center_y': .5})
        c1.add_widget(SentenceIn)


        CompleteSentenceLabelPre = Label(
            text='Most recently completed Sentence: ',
            size_hint=(.5, .8),
            pos_hint={'center_x': .22, 'center_y': .7})
        c1.add_widget(CompleteSentenceLabelPre)
        CompleteSentenceLabelPro = Label(
            text=self.RecentlyCompleted,
            size_hint=(.5, .8),
            pos_hint={'center_x': .22, 'center_y': .65})
        c1.add_widget(CompleteSentenceLabelPro)

        BorderLabel = Label(
            text='Instructions: Click on the text box above and enter a sentence with a word missing. \n Mark the location of the missing word with ___ (3 underscores), then press the submit button. \n \n Submitting no text will cause nothing to happen.',
            size_hint=(.5, .8),
            pos_hint={'center_x': .48, 'center_y': .2})
        c1.add_widget(BorderLabel)

        MisWordLabelPre = Label(
            text='Most recently predicted Word: ',
            size_hint=(.5, .8),
            pos_hint={'center_x': .72, 'center_y': .7})
        c1.add_widget(MisWordLabelPre)
        MisWordProLabel = Label(
            text=self.MissingWord,
            size_hint=(.5, .8),
            pos_hint={'center_x': .87, 'center_y': .7})
        c1.add_widget(MisWordProLabel)


        def UpdateText(instance):
            MisWordProLabel.text = self.MissingWord
            CompleteSentenceLabelPro.text = self.RecentlyCompleted
            CompleteSentenceLabelPro.text.replace(self.MissingWord, '\033[2;31;43m' + self.MissingWord)
            return True


        def callback(instance):
            FindMissingWord(SentenceIn.text)

        ButtonThing = Button(
                background_normal=AppFolderPath + "\ApplicationResources\SubmitButtonDesignOne.png",
                background_down=AppFolderPath + "\ApplicationResources\SubmitButtonDesignOneDown.png",
                size_hint=(.3, .10),
                pos_hint={'center_x': .48, 'center_y': .35})

        c1.add_widget(ButtonThing)

        ButtonThing.bind(on_press=callback)

        eventClock = Clock.schedule_interval(UpdateText, 2)

        return root 
        # Returns the "Root"* of the program. 
        # The Root, Contains all the "Widgets"* and GUI Instances in the program.
        # And behaves similarly to how a "Tree"* works.


        # *Root - The source of the program which all the other instances derive from.

        # *Widget - A Widget is an interface instance that is used to display, interact, take input, etc. from the User.

        # *Tree - A hierarchy which contains a source (Highest in the hierarchy) and "Children"* which derives from it, Seeing the "Root" as it's parent.

        # *Child - An instance which resides within another instance, object, etc. Which are higher in the hierarchy then the child itself.


if __name__ == '__main__':
    FindingWordo().run() 
    # Runs the application window according to the "MainAppClass" (Named FindingWordo) which has been built and configured above.