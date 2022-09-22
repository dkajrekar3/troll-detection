from copyreg import pickle
from requests import request
from flask import Flask, render_template ,request
import pickle
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
app = Flask(__name__)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



def transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)



@app.route("/" ,methods=['GET','POST'])
def home():
    if request.method == 'POST':
        input_message= request.form.get('message')
        transformed_message = transform(input_message)
        vector_message = tfidf.transform([transformed_message])
        result = model.predict(vector_message)[0]


        return render_template('index.html',result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)