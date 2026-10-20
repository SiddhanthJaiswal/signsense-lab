"""Phrase-level ASL plans, separate from learned motion and free-form generation."""
import re

GREETING_SOURCE = 'https://www.lifeprint.com/asl101/pages-signs/h/how.htm'

def phrase_plan(text):
    words = re.findall(r'[a-z]+', text.casefold())
    greeting = bool(words and words[0] in {'hi', 'hello', 'hey'})
    tail = words[1:] if greeting else words
    if tail == ['how', 'are', 'you'] or tail == ['how', 'you']:
        return {'glosses': (['hello'] if greeting else []) + ['how', 'you'],
                'source': GREETING_SOURCE, 'name': 'Friendly greeting'}
    if greeting and not tail:
        return {'glosses': ['hello'], 'source': GREETING_SOURCE, 'name': 'Greeting'}
    return None
