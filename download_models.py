import spacy

# Download the required models
spacy.cli.download("en_core_web_sm")
spacy.cli.download("nl_core_news_sm")