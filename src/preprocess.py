import re
def clean_text(text):
    text = text.lower()
    # remove urls
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # remove non-alphanumeric
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
