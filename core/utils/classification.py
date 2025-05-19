# classification.py
from transformers import pipeline
import re

# Базовые настройки
WAR_KEYWORDS = {
    'годы': ['1941', '1942', '1943', '1944', '1945', 'война', 'фронт'],
    'тематика': ['родина', 'победа', 'солдат', 'окопы', 'враг'],
    'авторы': ['сурков', 'лебедев-кумач', 'фатьянов']
}


def classify_war_song(text):
    try:
        # Попытка использовать модели
        ner = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl")
        entities = ner(text)
        authors = [e['word'] for e in entities if e['entity'] == 'PER']
    except:
        # Резервный вариант по ключевым словам
        authors = [word for word in WAR_KEYWORDS['авторы'] if word in text.lower()]

    # Извлечение года
    year = re.findall(r'\b(19[3-4][0-9])\b', text)
    year = max(year) if year else "не определен"

    # Определение тем
    themes = [
        theme for theme, keywords in WAR_KEYWORDS['тематика'].items()
        if any(kw in text for kw in keywords)
    ]

    return {
        'year': year,
        'themes': themes[:3],
        'authors': authors
    }