import os
from speach import transcribir
from nlp.elbert import load_weights, clasifySentiment
from argparse import Namespace

def main():
    path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(path, 'models/v1.1/ElBERT_weights.pth')
    model_sentiment = load_weights(full_path)
    trans_args = Namespace(model="base", non_english= True, energy_threshold= 1000, record_timeout=3, phrase_timeout= 3, default_microphone= "pulse")

    msg = ''
    for text in transcribir(trans_args):
        print(f'\r{text}', end='')
        msg += text
    
    print(msg)
    if not msg.strip():
        print("No se ha reconocido nada")
        exit()
    sentiment = clasifySentiment(model_sentiment, msg)
    print(model_sentiment.label_to_text(sentiment))
    print(sentiment)

if __name__ == '__main__':
    main()