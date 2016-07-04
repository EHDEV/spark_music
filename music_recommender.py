from pyspark.mllib.recommendation import ALS

def tokenize(line, exc_idx=None):
    """
    Method used to cast ids to integer and return tokens
    """
    tokens = ()
    for idx, tk in enumerate(line):
        try:
            if idx != exc_idx:
                tokens+=(int(tk),)
            else:
                tokens+=(tk,)
        except ValueError:
                tokens+=(0,"") if exc_idx else (0,)
    if len(tokens) > 2 and exc_idx:
        tokens = tokens[0:2]
    return tokens
def train_tokenize(row):
    """
    Prepares data for modeling
    """
    fin_artist_id = brdc_artists_alias.value.get(row[1]) \
        if brdc_artists_alias.value.get(row[1]) \
        else row[1]
    return (row[0], fin_artist_id, row[2])

# Loading artist data, artist alias and user artist data files into RDDs
    
raw_artist_data = sc.textFile("s3://ehec2/audio_data/artist_data.txt", 5)
artist_by_id = raw_artist_data.map(lambda line: line.split('\t'))\
                        .map(lambda token: tokenize(token, 1))
raw_artist_alias = sc.textFile("s3://ehec2/audio_data/artist_alias.txt")
artists_alias = raw_artist_alias.map(lambda line: line.split("\t"))\
                        .map(lambda token: tokenize(token))
raw_user_artist_data = sc.textFile("s3://ehec2/audio_data/user_artist_data.txt")
user_artist_data = raw_user_artist_data.map(lambda line: line.split())\
                        .map(lambda token: tokenize(token))

# Broadcast global variable for use in getting the right artist id
brdc_artists_alias = sc.broadcast(artists_alias.collectAsMap())

trainRDD = user_artist_data.map(lambda row: train_tokenize(row)).cache()
model = ALS.trainImplicit(trainRDD, 10, 5, 0.01, 1)

# Spot Checking Recommendations
# Extract the IDs of artists that this user has listened to and print their names. This means searching the input for artist IDs for this user, and then filtering the set of artists by these IDs so you can collect and print the names in order:
# Testing the model by looking at recommendations for user: 1052043

# To check recommendation for other users, change user_id here.
test_user_id = 2093760

# Getting all artists user test_user_id has listened to
artists_for_user = user_artist_data.filter(lambda l: l[0] == test_user_id)
existing_products = set(artists_for_user.map(lambda l: l[1]).collect())

# Getting the names and ids of artists user has listened to
artists_for_user = artist_by_id.filter(lambda art: art[0] in existing_products)\
                    .map(lambda art: art[1]).collect()

# print(artists_for_user)

# Getting top 10 recommendations for the user test_user_id. 

recommendations = model.call("recommendProducts",test_user_id, 10)
recommended_product_ids = set([r[1] for r in recommendations])

# Getting the recommended artist names.
recommended_artists = artist_by_id.filter(lambda art: art[0] in recommended_product_ids)\
                          .map(lambda tup: tup[1]).collect()

print("Top 10 Recommendations:\n")
for idx, muc in enumerate(recommended_artists):
    print ("(%i). %s"%(idx+1,muc))