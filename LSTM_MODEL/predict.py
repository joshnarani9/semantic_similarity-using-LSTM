import pandas as pd

import tensorflow as tf

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# File paths
TEST_CSV = './data/Text_Similarity_Dataset.csv'

# Load training set
test_df = pd.read_csv(TEST_CSV)
test_df=test_df.rename(columns={"text1":"question1","text2":"question2"})
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)

    
# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --loading model

model = tf.keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
model.summary()
#predicting the similarity scores
prediction = model.predict([X_test['left'], X_test['right']])
print(prediction)
#As the prediction values are numpy.ndarray,so converting to list and saving into csv file
pr=[]
for i in range(0,len(prediction)):
    pr.append(prediction[i][0])
#pr=prediction[1][0]
#pr=pd.Series(prediction[0])
values=pd.DataFrame({'Unique_ID':test_df['Unique_ID'],'Similarity_Score':pr})
values.to_csv('similarityscores.csv')

######################saving test and embedding values into files
test.to_csv('./data/testvalues.csv')

em=[]
for i in range(0,len(embeddings)):
    em.append(embeddings[i][0])
embedding=pd.DataFrame({'embeddings':em})
embedding.to_csv('./data/embeddingvalues.csv')

##########assigning scoring values which are less than 0.6 with 0 and remaining with 1#####
import pandas as pd
dff=pd.read_csv('similarityscores.csv')
sim=dff['Similarity_Score']

def sc(sc):
    if sc<0.6:
        return 0
    else:
        return 1
dff['Similarity_Score']=dff['Similarity_Score'].apply( lambda x:sc(x))
pr=dff['Similarity_Score'].tolist()

values=pd.DataFrame({'Unique_ID':test_df['Unique_ID'],'Similarity_Score':pr})
values.to_csv('similarityscores.csv')
dff['Similarity_Score'].value_counts()