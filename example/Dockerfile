FROM tensorflow/tensorflow

ENV TFHUB_CACHE_DIR /tfhub_modules
WORKDIR /app/
ADD requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

ADD extract_similar_sentences.py /app/extract_similar_sentences.py
