# model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def create_model(data_hotel, data_kunjungan):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_amenities = tfidf_vectorizer.fit_transform(
        data_hotel['Amenities'])

    scaler = MinMaxScaler()
    scaled_numerical_data = scaler.fit_transform(
        data_hotel[['Rating', 'Price', 'Latitude', 'Longitude']])

    category_dummies = pd.get_dummies(data_hotel['Category'])

    combined_features = pd.concat([pd.DataFrame(tfidf_matrix_amenities.toarray()),
                                   pd.DataFrame(scaled_numerical_data),
                                   category_dummies.reset_index(drop=True)], axis=1)

    combined_features.index = data_hotel['no_id']

    return combined_features


def recommend_hotels(user_id, data_hotel, data_kunjungan, combined_features):
    user_hotels = data_kunjungan[data_kunjungan['no_user'] == user_id]

    if user_hotels.empty:
        return "User has no hotel visits recorded."

    user_profiles = combined_features.loc[user_hotels['no_id']].mean(axis=0)

    cosine_similarities = cosine_similarity([user_profiles], combined_features)

    similarity_scores = list(enumerate(cosine_similarities[0]))
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_hotels = []
    for idx, score in similarity_scores:
        hotel_id = combined_features.index[idx]
        if hotel_id not in user_hotels['no_id'].values:
            recommended_hotels.append(
                (data_hotel[data_hotel['no_id'] == hotel_id]['Hotel'].values[0], score))
        if len(recommended_hotels) >= 5:
            break

    return recommended_hotels


def recommend_similar_hotels(hotel_name, data_hotel, combined_features):
    if hotel_name not in data_hotel['Hotel'].values:
        return "Hotel not found."

    item_index = data_hotel[data_hotel['Hotel'] == hotel_name].index[0]
    cosine_similarities_content = cosine_similarity(
        combined_features, combined_features)

    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:10]

    recommended_items_indices = [x[0] for x in top_similar_items]

    return data_hotel.iloc[recommended_items_indices][['Hotel', 'Category', 'Rating', 'Price', 'Amenities', 'Latitude', 'Longitude']]
