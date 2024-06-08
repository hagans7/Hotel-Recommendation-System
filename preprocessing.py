    # preprocessing.py
    import pandas as pd


    def load_and_preprocess_data(hotel_path, kunjungan_path):
        data_hotel = pd.read_csv(hotel_path)
        data_kunjungan = pd.read_csv(kunjungan_path)

        # Remove 'Rp' and convert 'Price' to numeric
        data_hotel['Price'] = data_hotel['Price'].str.replace(
            'Rp', '').str.replace(',', '').astype(float)
        data_kunjungan['Price'] = data_kunjungan['Price'].str.replace(
            'Rp', '').str.replace(',', '').astype(float)

        data_hotel = data_hotel.dropna()
        data_kunjungan = data_kunjungan.dropna()

        return data_hotel, data_kunjungan
