import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBased_mean:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, data):
        self.clicks = data['clicks']
        self.articles = data['articles']
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def get_user_profil(self, user):
        user_clicks = self.clicks[self.clicks['user_id'] == user]['click_article_id'].tolist()
        user_catchy = self.articles.loc[user_clicks]
        user_profil = user_catchy.groupby(lambda x: user).mean()
        user_profil.drop(columns=['article_id'], inplace = True)
        return user_profil

    def get_reco_from_profile(self, profile):
        # get the  similarity scores sorted with that article
        to_sim = self.articles.drop(columns=['article_id'])
        article_sim_scores = cosine_similarity(profile, to_sim)
        article_sim_scores = [i for i in enumerate(article_sim_scores[0])]
        article_sim_scores = sorted(list(article_sim_scores), key=lambda x: x[1], reverse=True)
        article_sim_scores = article_sim_scores[1:]
        
        #get the indices & scores
        reco_indices = [i[0] for i in article_sim_scores]
        scores = [i[1] for i in article_sim_scores]        
        
        # return as df
        reco_from_one_df = pd.DataFrame()
        reco_from_one_df['article_id'] = reco_indices
        reco_from_one_df['sim_scores'] = scores
        reco_from_one_df['ranking'] = range(1, len(reco_from_one_df) + 1)
    
        return reco_from_one_df  
        
    def recommend_items(self, user, nb = 10):
        
        profile = self.get_user_profil(user)      
        user_items = self.clicks[self.clicks['user_id'] == user]['click_article_id'].tolist()
        
        suggestions = self.get_reco_from_profile(profile)
        
        suggestions_filtered = suggestions[~suggestions['article_id'].isin(user_items)].reset_index()
        reco = pd.DataFrame(suggestions_filtered, columns=['article_id', 'sim_scores']).head(nb)
        #reco = reco.merge(self.articles['category_id'], how = 'left', left_on = 'article_id', right_index = True)

        return reco

class CollaborativeFiltering:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_preds_df, data):
        self.predictions = cf_preds_df
        self.articles = data['articles']
        self.clicks = data['clicks']
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user, nb = 10):     
        
        user_items = self.clicks.loc[user]['click_article_id'].tolist()
        
        sorted_predictions = self.predictions[user].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user: 'strength'})

        reco = sorted_predictions[~sorted_predictions['click_article_id'].isin(user_items)] \
                                .sort_values('strength', ascending = False) \
                                .reset_index().drop(columns=['index']).head(nb)
        reco.rename(columns={'click_article_id': 'article_id'}, inplace = True)
        reco = reco.merge(self.articles['category_id'], how = 'left', left_on = 'article_id', right_index = True)

        return reco

class Hybrid:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_model, cf_model, data, cb_weight = 1.0, cf_weight = 1.0):
        self.cb = cb_model
        self.cf = cf_model
        self.cb_weight = cb_weight
        self.cf_weight = cf_weight
        self.articles = data['articles']
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user, nb = 10):
               
        NB = 100 * nb
        #Getting the top Content-based filtering recommendations
        cb_reco_df = self.cb.recommend_items(user, nb = NB) \
        .rename(columns = {'sim_scores': 'strength_CB'}).drop(columns = ['category_id'])
        #cb_reco_df = normalisation_reco(cb_reco_df)
        
        #Getting the top Collaborative filtering recommendations
        cf_reco_df = self.cf.recommend_items(user, nb = NB) \
        .rename(columns = {'strength': 'strength_CF'}).drop(columns = ['category_id'])
        #cf_reco_df = normalisation_reco(cf_reco_df)
        
        #Combining the results by contentId
        reco_df = cb_reco_df.merge(cf_reco_df,
                                   how = 'outer', 
                                   left_on = 'article_id', 
                                   right_on = 'article_id').fillna(0.0)
        
        #Computing a hybrid recommendation score based on CF and CB scores
        #recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF'] 
        reco_df['strength_Hybrid'] = (reco_df['strength_CB'] * self.cb_weight) \
                                     + (reco_df['strength_CF'] * self.cf_weight)
        
        #Sorting recommendations by hybrid score
        reco = reco_df.sort_values('strength_Hybrid', ascending = False) \
        .reset_index().drop(columns=['index']).head(nb)
        reco = reco.merge(self.articles['category_id'], how = 'left', left_on = 'article_id', right_index = True)
     
        return reco