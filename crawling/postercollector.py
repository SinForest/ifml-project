#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:57:17 2018

@author: twuensche
"""

import sys
from bs4 import BeautifulSoup
import requests
import pickle
import os
from urllib.request import urlopen
from random import randint
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

processes = None

headers = {
    'headers': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'}
test_ids = ['tt0438097', 'tt0096895', 'tt0111161', 'tt0167260', 'tt0068646',\
            'tt0071562', 'tt0468569', 'tt0050083', 'tt0108052', 'tt0110912',\
            'tt0060196']
genres = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime',\
          'documentary', 'drama', 'family', 'fantasy', 'film-noir', 'history',\
          'horror', 'music', 'musical', 'mystery', 'romance', 'sci-fi',\
          'short', 'sport', 'superhero', 'thriller', 'war', 'western']
timeout_param = 3
autosave_interval = 20

def main():
    mode, inputs = handle_user_input()
    if mode == 'all':
        download_movie_list_multiprocess(inputs)
        download_movie_posters_multiprocess()        
    elif mode == 'list':
        download_movie_list_multiprocess(inputs)
    elif mode == 'download':
        download_movie_posters_multiprocess()




def scrape_from_imdb_id(id):
    #The actual query
    url = "https://www.imdb.com/title/" + id
    title = '-'
    genres = []
    poster = []
    record = {}

    try:
        url = url.rstrip('\n')
        r = requests.get(url, headers=headers, timeout = timeout_param)
        if r.status_code != 200:
            return None

        html = r.text
        soup = BeautifulSoup(html, 'lxml')
        title_section = soup.select('.title_wrapper > h1')
        genre_list = soup.find("div", id="titleStoryLine")
        poster_section = soup.select('.poster')

        if title_section:
            title = title_section[0].text.strip()
        if genre_list:
            found_genres = genre_list.find(itemprop='genre').findAll('a')
            
            for genre in found_genres:
                genre_text = str.strip(genre.text)
                genres.append(genre_text)
        if poster_section:
            poster = poster_section[0].img['src']

        record = {'imdb-id': id, 'title': title, 'genres': genres, 'poster': poster}
        if not (id == '-' or genres == [] or poster == []):
            return record
        else:
            return None
    except Exception as ex:
        pass
    
def handle_user_input():
    inputs = sys.argv[1:]
    mode = ''
    if len(inputs) > 0:
        if inputs[0] == 'test':
            inputs = test_ids
            mode = 'all'
        elif inputs[0] == 'testlist':
            inputs = test_ids
            mode = 'list'
        elif inputs[0] == 'randomMovies':
            inputs = random_ids(inputs[1])
            mode = 'list'
        elif inputs[0] == 'topMovies':
            inputs = top_movies()
            mode = 'list'
        elif inputs[0] == 'byGenre':
            user_genre = inputs[1]
            pages = int(inputs[2])        
            if len(inputs) > 3:
                global processes
                processes = int(inputs[3])
            inputs = []
            if user_genre == 'all':
                for genre in genres:
                    print('getting ids for ' + genre)
                    inputs.extend(movies_by_genre_multiprocess(genre, pages))
            else:
                inputs.extend(movies_by_genre_multiprocess(user_genre, pages))
            mode = 'list'
        elif inputs[0] == 'all':
            inputs = inputs[1:]
            mode = 'all'
        elif inputs[0] == 'list':
            inputs = inputs[1:]
            mode = 'list'
        elif inputs[0] == 'download':
            if len(inputs) > 1:
                global processes
                processes = int(inputs[1])
            inputs = inputs[2:]
            mode = 'download'
    else:
        help()
        sys.exit()
    return mode, inputs

def download_movie_list(ids):
    movies = read_movies()
    existing_ids = []
    autosave_counter = 0
    for movie in movies:
        existing_ids.append(movie['imdb-id'])
        
    for id in ids:
        if not id in existing_ids:
            movie = scrape_from_imdb_id(id)
            existing_ids.append(id)
            if not movie is None:
                movies.append(movie)
                autosave_counter += 1
                if (autosave_counter == autosave_interval):
                    autosave_counter = 0
                    write_movies(movies)
        else:
            print(id + ' already exists')
    write_movies(movies)
    
def download_movie_list_multiprocess(ids):
    print('downloading movie list')
    pool = Pool(processes)
    movies = read_movies()
    existing_ids = []
    autosave_counter = 0
    for movie in movies:
        existing_ids.append(movie['imdb-id'])
    
    clean_ids = list(set(ids) - set(existing_ids))
    
    for movie in tqdm(pool.imap(scrape_from_imdb_id, clean_ids)):
        if not movie is None:
            movies.append(movie)
            autosave_counter += 1
            if (autosave_counter == autosave_interval):
                autosave_counter = 0
                write_movies(movies)
    write_movies(movies)
    
def download_movie_posters():
    movies = read_movies()
    if not os.path.exists('posters'):
        os.makedirs('posters')
    for movie in movies:
        filename = 'posters/' + movie['imdb-id'] + ".jpg"
        if not os.path.exists(filename):
            print('Downloading poster for ' + movie['title'])
            
            try:   
                request = urlopen(movie['poster'], timeout = timeout_param)
                with open(filename, 'wb') as f:
                    f.write(request.read())            
            except Exception as ex:
                print(str(ex))
            
        else:
            print('Poster for ' + movie['title'] + ' already exists')

def download_movie_posters_multiprocess():
    print('downloading posters')
    pool = Pool(processes)
    movies = read_movies()
    if not os.path.exists('posters'):
        os.makedirs('posters')
    for success in tqdm(pool.imap(download_poster, movies)):
        pass
            
def download_poster(movie):
    filename = 'posters/' + movie['imdb-id'] + ".jpg"
    if not os.path.exists(filename):
        try:   
            request = urlopen(movie['poster'], timeout = timeout_param)
            with open(filename, 'wb') as f:
                f.write(request.read())
                return True            
        except Exception as ex:
            return False

def random_ids(num):
    num = int(num)
    ids = []
    for i in range(num):
        id = 'tt' + str(randint(0,8500000)).zfill(7)
        if id not in ids:
            ids.append(id)
        else:
            i-=1
    return ids

def top_movies():
    ids=[]
    try:
        base_url = 'http://www.imdb.com'
        url = base_url + '/chart/top'
        r = requests.get(url, headers=headers)
        html = None
        if r.status_code != 200:
            return None
        html = r.text
        soup = BeautifulSoup(html, 'lxml')
        titles = soup.select('.titleColumn a')
        if titles is not None:
            for title in titles:
                id = title['href']
                id = id[7:16]
                ids.append(id)
    except Exception as ex:
        print(str(ex))
    finally:
        return ids

def movies_by_genre(genre, pages):
    ids=[]
    for page in range(1,pages+1):
        try:   
            print('page ' + str(page) + '..')
            base_url = 'http://www.imdb.com'
            url = base_url + '/search/title?genres=' + genre + '&page=' + str(page)
            r = requests.get(url, headers=headers, timeout = timeout_param)
            html = None
            if r.status_code != 200:
                return None
            html = r.text
            soup = BeautifulSoup(html, 'lxml')
            titles = soup.select('.lister-item-header a')
            if titles is not None:
                for title in titles:
                    id = title['href']
                    id = id[7:16]
                    ids.append(id)
            else:
                print('A problem occurred, skipping this page...')
        except Exception as ex:
            print(str(ex))
        
        
    return ids

def movies_by_genre_multiprocess(genre, pages):
    pool = Pool(processes)
    ids=[]
    pages = list(range(1,pages+1))
    func = partial(ids_by_page, genre)
    for page_ids in tqdm(pool.imap(func, pages)):
        ids.extend(page_ids)
    return ids

def ids_by_page(genre, page):
    ids = []
    try:   
        base_url = 'http://www.imdb.com'
        url = base_url + '/search/title?genres=' + genre + '&page=' + str(page)
        r = requests.get(url, headers=headers, timeout = timeout_param)
        html = None
        if r.status_code != 200:
            return None
        html = r.text
        soup = BeautifulSoup(html, 'lxml')
        titles = soup.select('.lister-item-header a')
        if titles is not None:
            for title in titles:
                id = title['href']
                id = id[7:16]
                ids.append(id)
        else:
            pass
    except Exception as ex:
        pass
    
    return ids

def write_movies(movies):
    with open('movielist', 'wb') as fp:
        pickle.dump(movies, fp)
        
        
def read_movies():
    if os.path.exists('movielist'):
        with open ('movielist', 'rb') as fp:
            return pickle.load(fp)
    else:
        return []

def help():
    print('How to use the postercollector:')
    print('The postercollector works in two stages. In the list stage,' + \
          ' imdb-ids, titles, genres and poster urls are downloaded and' + \
          ' stored in the movielist file in the active folder. This list' + \
          ' is cummulative, movies are added to the existing list, nothing' + \
          ' gets deleted or replaced. If a movie times out, running the same'+ \
          ' command again will likely fix the problem (existing movies will' + \
          ' not be added again).')
    print('In the download stage, posters for existing movies are downloaded.' + \
          ' These posters will be named by the movies imdb-id and stored'+ \
          ' in a folder called posters in the active directory.')
    print('To add to the movie list call the script with the argument' + \
          ' \"topMovies\" to add the 250 highest rated movies.')
    print('Use the argument \"byGenre\" followed by a the name of the genre' + \
          ' and the number of pages (50 movies per page) to add movies from' + \
          ' a specific genre.')
    print('Example: $python postercollector.py byGenre all 2')
    print('This command will add 100 movies from each genre to the list.')
    print('Use the argument \"download\" to download the posters of all' + \
          'movies in the list (unless their poster is already present).')
    
if  __name__ =='__main__':main()