from bs4 import BeautifulSoup
import requests
import json
from zipfile import ZipFile

#Function for getting a particular chapter of Mahabharat
def get_chapter(chapter_url):
    
    page = requests.get(chapter_url)

    if page.status_code == 200:
      soup=BeautifulSoup(page.content,'html5lib')

      elements = soup.find_all('td')[0]

      for br in elements.find_all("br"):
          br.replace_with("")
      for br in elements.find_all("font"):
          br.replace_with("")

      chapter = []
      for i in elements:
          chapter.append(i.strip())
      chapter = [i for i in chapter if i != '']
      return chapter

    else:
        return 0

#Storing mahabharat as json file tagged by book number and chapter

def mahabharat_json():
    
    mahabharat = {}
    
    for book in range(1,19):
      book_num = "%02d" % book
      for chapter in range(1, 500):
        chapter_num = "%03d" % chapter
        url = "https://www.sacred-texts.com/hin/mbs/mbs" + book_num+chapter_num + ".htm"
        if book_num not in mahabharat:
          mahabharat[book_num] = {}
        mahabharat[book_num][chapter_num] = get_chapter(url)

    with open('mahabharat.json', 'w') as fp:
        json.dump(mahabharat, fp)

    return 0
    
    
#Downloading books and storing them as text files

def mahabharat_text():

    mahabharat = {}

    for book in range(1,19):
      book_content = []
      book_num = "%02d" % book
      for chapter in range(1, 500):
        chapter_num = "%03d" % chapter
        url = "https://www.sacred-texts.com/hin/mbs/mbs" + book_num+chapter_num + ".htm"
        content = get_chapter(url)
        if type(content) == list:
          book_content.extend(get_chapter(url))
      filename = 'book' + book_num + '.txt'
      with open(filename, 'w') as f:
        for item in book_content:
            f.write("%s\n" % item)

    #Create a ZipFile Object
    with ZipFile('mahabharat.zip', 'a') as zipObj:
       # Add multiple files to the zip
      for num in range(1,19):
        book_num = "%02d" % num
        filename = 'book' + book_num + '.txt'
        zipObj.write(filename)
        
    return 0