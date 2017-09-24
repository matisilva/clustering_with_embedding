import spacy
nlp = spacy.load('es_core_web_md')
doc = nlp(u'Batman colaboró con todos en Nicaragua y perdió el sueldo en el casino Royal')
for token in doc:
    print(token.ent_type_, token.dep_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])
