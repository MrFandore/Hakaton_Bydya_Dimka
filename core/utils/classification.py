from transformers import pipeline
import re

WAR_KEYWORDS = {
    'годы': ['1941', '1942', '1943', '1944', '1945', 'война', 'фронт'],
    'тематика': ['родина', 'победа', 'солдат', 'окопы', 'враг'],
    'авторы': ['сурков', 'лебедев-кумач', 'фатьянов']
}


def classify_war_song(text):
    try:
        ner = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl")
        entities = ner(text)
        authors = [e['word'] for e in entities if e['entity'] == 'PER']
    except:
        authors = [word for word in WAR_KEYWORDS['авторы'] if word in text.lower()]
    themes = [
        theme for theme, keywords in WAR_KEYWORDS['тематика'].items()
        if any(kw in text for kw in keywords)
    ]

    return {
        'year': year,
        'themes': themes[:3],
        'authors': authors
    }
