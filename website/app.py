
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

model_path = "./bert-large-model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)

def split_sentence(sentences):
    new_sentence = sentences.replace('!', '!.')
    new_sentence = new_sentence.replace('?', '?.')

    sentence_arr = new_sentence.split('.')
    return sentence_arr

def predict_sentiment(sentence_arr):
    class_names = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    sentence_results = []
    combined_probabilities = None
    valid_count = 0

    for sentence in sentence_arr:
        if not sentence: # Skip empty sentences
            continue
        
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Compute probabilities for the sentence
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        
        # Append result for the current sentence
        sentence_results.append({
            "sentence": sentence,
            "sentiment": class_names[predicted_label],
            "probabilities": probabilities.cpu().numpy().tolist()[0]
        })
        
        # Accumulate probabilities for overall sentiment calculation
        if combined_probabilities is None:
            combined_probabilities = probabilities
        else:
            combined_probabilities += probabilities
        valid_count += 1

    # Averege the probabilities for final sentiment
    if valid_count > 0:
        average_probabilities = combined_probabilities / valid_count
        final_label = torch.argmax(average_probabilities, dim=1).item()
        final_sentiment = class_names[final_label]
    else:
        final_sentiment = "Neutral" # Default sentiment (No valid sentiment)
    
    return sentence_results, final_sentiment


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sentences = data.get("sentence", "")
    sentence_arr = split_sentence(sentences)
    sentence_sentiments, final_sentiment = predict_sentiment(sentence_arr)
    return jsonify({"sentence_sentiments": sentence_sentiments, "final_sentiment": final_sentiment})

if __name__ == "__main__":
    # socketio.run(app, host='0.0.0.0', port=5001, debug=True)
    app.run(debug=True)
