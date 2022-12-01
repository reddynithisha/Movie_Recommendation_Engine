from django.http import JsonResponse
from django.shortcuts import render
from django.http.response import HttpResponse
import pandas as pd
import numpy as np
from json import dumps
import pickle
from tmdbv3api import TMDb
import json
import requests
from gensim.models import KeyedVectors
tmdb = TMDb()
tmdb.api_key = 'Your AP1 Key'
from tmdbv3api import Movie
tmdb_movie = Movie()
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import bs4
import urllib
import bz2file as bz2
from keras.models import load_model
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from gensim.utils import simple_preprocess


data = pd.read_csv(r"C:\Users\nitisha.reddy\movie_recommendation\recommendation\Datasets\Movie_main_data_final.csv")
mat = CountVectorizer()
count_matrix = mat.fit_transform(data['comb'])

vector= bz2.BZ2File(r"C:\Users\nitisha.reddy\movie_recommendation\recommendation\Datasets\word2vec_bz2file_decimal.pbz2", "rb")
vector=pickle.load(vector)

ann = load_model(r'C:\Users\nitisha.reddy\movie_recommendation\recommendation\Datasets\ann.h5')

lemmatizer = WordNetLemmatizer()

def cosinesimilarity(x):
    similarity = cosine_similarity(x)
    return similarity

def sentiment_analysis(z):    
    corpus1=[]
    for i in range(0,1):
        review = re.sub('[^a-zA-Z]', ' ', z)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus1.append(review)
        words2 = []
        for i in corpus1:
            sent_token = sent_tokenize(i)
            for j in sent_token:
                words2.append(simple_preprocess(j))
        x = np.array( np.mean([vector[word] for word in words2[0] if word in vector.index_to_key],axis=0))
        ans = ann.predict(x.reshape(1,100))
        ans = (ans > 0.5)
    return ans[0][0]

def home(request):
    movies = list(data['movie_title'])
    return render(request,'home.html',{'movie':movies})

def movie_details(request):
    query = request.POST.get('search')
    movie = tmdb_movie.search(query)
    if (len(movie)<1):
        mesg=1
        return render(request,'sample.html',{'mesg':mesg})
    else:
        mesg=0
        movie_id = movie[0].id
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        data_json = response.json()
        imdb_id = data_json['imdb_id']
        response1 = requests.get('https://api.themoviedb.org/3/movie/{}/credits?api_key={}'.format(movie_id,tmdb.api_key))
        credit_json = response1.json()
        Overview  = data_json['overview']
        Poster    =  data_json['poster_path']
        Poster    =  "https://image.tmdb.org/t/p/original"+Poster
        Tagline   = data_json['tagline']
        Release_date= data_json['release_date']
        Vote_average = data_json['vote_average']
        Budget = data_json['budget']
        Revenue = data_json['revenue']
        director = data[data['movie_title']==query]['director_name']
        director = str(director)[:-34]
        b = data[data['movie_title']==query].index[0]
        cast1 = credit_json['cast'][0]['original_name']
        char1 = credit_json['cast'][0]['character']
        cast_profile1 = credit_json['cast'][0]['profile_path']
        cast2 = credit_json['cast'][1]['original_name']
        char2 = credit_json['cast'][1]['character']
        cast_profile2 = credit_json['cast'][1]['profile_path']
        cast3 = credit_json['cast'][2]['original_name']
        char3 = credit_json['cast'][2]['character']
        cast_profile3 = credit_json['cast'][2]['profile_path']
        cast_profile1 = "https://image.tmdb.org/t/p/original"+cast_profile1
        cast_profile2 = "https://image.tmdb.org/t/p/original"+cast_profile2
        cast_profile3 = "https://image.tmdb.org/t/p/original"+cast_profile3
        cast_profile = [cast_profile1 , cast_profile2, cast_profile3]
        cast_name = [cast1,cast2,cast3]
        char_name = [char1,char2,char3]
        cast = zip(cast_profile,cast_name,char_name)
        lst = list(enumerate(cosinesimilarity(count_matrix)[b]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        images = []
    
        for j in l:
            dum =  tmdb_movie.search(j)
            dum_id = dum[0].id
            resp = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(dum_id,tmdb.api_key))
            da_json = resp.json()
            images.append("https://image.tmdb.org/t/p/original"+da_json['poster_path'])


        sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
        soup = bs4.BeautifulSoup(sauce,'lxml')
        soup_result = soup.find_all("div",{"class":"text show-more__control"})
        soup_result = list(soup_result)
        review=[]
        sentiment = []
        if len(soup_result)>=10: 
            for i in range(10):
                review.append(str(soup_result[i])[37:-7])
                senti = str(soup_result[i])[37:-7]
                senti = sentiment_analysis(senti)
                if senti:
                    sentiment.append('Positive')
                else:
                    sentiment.append('Negative')       
        else: 
            for i in range(len(soup_result)):
                
                review.append(str(soup_result[i])[37:-7])   
                senti = str(soup_result[i])[37:-7]
                senti = sentiment_analysis(senti)
                if senti:
                   sentiment.append('Positive')
                else:
                   sentiment.append('Negative') 
        
        myreview = zip(review,sentiment)
        my_recommendations = zip(images,l)

        return render(request,'sample.html',{'mesg':mesg,'Movie':query,'Poster':Poster,'my_recommendations':my_recommendations,'overview':Overview,'tagline':Tagline,'release_date':Release_date,'vote_average':Vote_average,'budget':Budget,'revenue':Revenue,'director':director,'cast':cast,'myreview':myreview})

def autosuggestion(request):
    query = request.GET.get('term')
    results = []
    q = query.lower()
    for i in data['movie_title']:
        if str(i):
            if q in i.lower():
                results.append(i)

    return JsonResponse(results,safe=False)
