# I have taken help from kaggle kernals for tokenizing the the item description and label encoding using cat.codes
import pandas as pd
import numpy as np
import regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor as RFR

# reading files
train=pd.read_csv("../input/train.tsv",sep="\t",engine="python")
test=pd.read_csv("../input/test.tsv",sep="\t",engine="python")
# converting price into log because it is skewed
train["log_price"]=np.log(train["price"]+1)
# splitting the category_name into sub category
def split_feature(text):
    return text.split("/")
# splitting the train.Category_name    
train['general_cat'], train['subcat_1'], train['subcat_2'] = \
zip(*train['category_name'].apply(lambda x: split_feature(x)))
# splitting the test.Category_name 
test['general_cat'], test['subcat_1'], test['subcat_2'] = \
zip(*test['category_name'].apply(lambda x: split_feature(x))) 
# defining function for tokenizing
def wordCount(text):
    try:
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words and len(w)>3]
        return len(words)
    except:
        return 0
# applying the wordcount on both set
train['desc_len'] = train['item_description'].apply(lambda x: wordCount(x))
test['desc_len'] = test['item_description'].apply(lambda x: wordCount(x))
# removing the 4 null item description rows
train = train[pd.notnull(train['item_description'])]
# defining the tokenize function 
def tokenize(text):
    try: 
        stop_words = set(stopwords.words('english'))
        regex1 = regex.compile('[' +regex.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex1.sub(" ", text) 
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop_words, tokens))
        filtered_tokens = [w for w in tokens if regex.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        return filtered_tokens
    except TypeError as e: print(text,e)
# getting the tfidf matrix        
vectorizer = TfidfVectorizer(min_df=10,
                             max_features=180000,
                             tokenizer=tokenize,
                             ngram_range=(1, 2))        
all_items = np.append(train['item_description'].values, test['item_description'].values)
vz = vectorizer.fit_transform(list(all_items))
svd = TruncatedSVD(n_components=4, random_state=42)
svd_tfidf = svd.fit_transform(vz)
train_test_tfidf=pd.DataFrame(svd_tfidf,columns=["pc1","pc2","pc3","pc4"]) # using the four LSA components as additional features
# label encoding the categorical variables
train['is_train'] = 1
test['is_train'] = 0
train_test_combine = pd.concat([train.drop(['price',"log_price","category_name","item_description"],axis =1),test.drop(["category_name","item_description"],axis=1)],axis = 0)
train_test_combine.name = train_test_combine.name.astype('category')
train_test_combine.brand_name = train_test_combine.brand_name.astype('category')
train_test_combine.general_cat = train_test_combine.general_cat.astype('category')
train_test_combine.subcat_1 = train_test_combine.subcat_1.astype('category')
train_test_combine.subcat_2 = train_test_combine.subcat_2.astype('category')
train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
train_test_combine.general_cat = train_test_combine.general_cat.cat.codes
train_test_combine.subcat_1 = train_test_combine.subcat_1.cat.codes
train_test_combine.subcat_2 = train_test_combine.subcat_2.cat.codes
# modeling
train_test_combine=train_test_combine.drop(["test_id","train_id"],axis=1)
train_test_combined=pd.concat([train_test_combine.reset_index(drop=True),train_test_tfidf.reset_index(drop=True)],axis=1)
df_train=train_test_combined.loc[train_test_combined["is_train"]==1]
df_test=train_test_combined.loc[train_test_combined["is_train"]==0]
df_test=df_test.drop(["is_train"],axis=1)
df_train=df_train.drop(["is_train"],axis=1)
df_train["log_price"]=train.log_price.values
x_train,y_train = df_train.drop(['log_price'],axis =1),df_train.log_price
model=RFR(n_estimators=4)
model.fit(x_train,y_train)
y_test=model.predict(df_test)
submission=pd.DataFrame({"test_id":list(test["test_id"])})
submission["price"]=y_test
submission["price"]=submission["price"].apply(lambda x:np.exp(x)-1)