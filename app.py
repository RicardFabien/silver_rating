from joblib import load
import gradio as gr
from sklearn.pipeline import make_pipeline

# Load the model
model_bayes = load('filename.joblib')

# Prediction function
def make_prediction(user_sentence):
  
  prediction = model_bayes.predict([user_sentence])
  dict = {0: 'Negative', 1: 'Positive'}
  return dict[prediction[0]]

title = "Sentiment Analysis movie Reviews"
description = "<p style='text-align: center'>Identifier si un commentaire sur un film est negatif ou positif<br/>Entrainer sur .</p>"
examples = ["The movie was...bad. The actors suck, the camerawork is shabby at best and the scenario is completly non-sensical", "This is amazing !"]

app = gr.Interface(fn=make_prediction, title=title, description=description, examples=examples, inputs=gr.TextArea(), outputs=gr.Label(num_top_classes=3))

app.launch(share=True)