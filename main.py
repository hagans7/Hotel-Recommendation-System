from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import List
from pydantic import BaseModel
from datetime import timedelta
from preprocessing import load_and_preprocess_data
from model import recommend_hotels, recommend_similar_hotels, create_model
from security import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    fake_users_db
)

app = FastAPI()

# Load and preprocess data
data_hotel, data_kunjungan = load_and_preprocess_data(
    'data-pakai/df_dataset-clean.csv', 'data-pakai/df_dataset-recom.csv')
combined_features = create_model(data_hotel, data_kunjungan)

# Token model


class Token(BaseModel):
    access_token: str
    token_type: str

# User model


class User(BaseModel):
    username: str
    email: str
    full_name: str = None
    disabled: bool = None


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(
        fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Hotel Recommendation System"}


@app.post("/recommendations/")
async def get_recommendations(user_id: int, current_user: User = Depends(get_current_active_user)):
    recommendations = recommend_hotels(
        user_id, data_hotel, data_kunjungan, combined_features)
    return recommendations


@app.post("/similar_hotels/")
async def get_similar_hotels(hotel_name: str, current_user: User = Depends(get_current_active_user)):
    similar_hotels = recommend_similar_hotels(
        hotel_name, data_hotel, combined_features)
    return similar_hotels.to_dict(orient="records")
