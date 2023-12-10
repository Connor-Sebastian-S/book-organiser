
from bs4 import BeautifulSoup
import requests
import numpy as np
import csv
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import shutil
import os
import pandas as pd
import urllib

LINK = "https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2334524.m570.l1313&_nkw=m2+macbook+pro&_sacat=0&LH_TitleDesc=0&_odkw=m1+macbook+air&_osacat=0"

# Machine Learning Model
model = LinearRegression()

# Backup file path
BACKUP_FILE = 'data_backup.csv'

def get_prices_by_link(link):
    # get source
    r = requests.get(link)
    # parse source
    page_parse = BeautifulSoup(r.text, 'html.parser')
    # find all list items from search results
    search_results = page_parse.find("ul", {"class":"srp-results"}).find_all("li", {"class":"s-item"})

    item_url = []
    for result in search_results:
        url = result.find("a", {"class":"s-item__link"})['href']
        #link = result.find('span', href=True)['href']
        #print(url)
        #url.get('href')
        #info=soup.find(‘a’, href=True)
        item_url.append(url)

    item_prices = []

    # find prices of all results
    for result in search_results:
        price_as_text = result.find("span", {"class":"s-item__price"}).text
        if "to" in price_as_text:
            continue
        # converting obtained string to a number
        price = float(price_as_text[1:].replace(",",""))
        item_prices.append(price)

    return item_prices, item_url

# if price is two S.D above or below the mean, then ignore the price
def remove_outliers(prices, m=2):
    data = np.array(prices)
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def get_average(prices):
    return np.mean(prices)

def get_lowest(prices):
    index = np.where(prices == np.amin(prices))
    return np.amin(prices), index

# save prices with date to a csv file
def save_to_file(prices):
    fields = [datetime.today().strftime("%B-%D-%Y"), np.around(get_average(prices), 2)]
    with open('prices.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def train_price_prediction_model(prices):
    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    return model

def predict_price(model, days):
    latest_price = get_average(remove_outliers(get_prices_by_link(LINK)))
    future_dates = np.arange(len(model.coef_), len(model.coef_) + days).reshape(-1, 1)
    future_prices = model.predict(future_dates)

    return latest_price, future_prices

def backup_data():
    shutil.copyfile('prices.csv', BACKUP_FILE)
    print("Data backup successful.")

def restore_data():
    if os.path.exists(BACKUP_FILE):
        shutil.copyfile(BACKUP_FILE, 'prices.csv')
        print("Data restoration successful.")
    else:
        print("No backup file found.")

if __name__ == '__main__':

    # Read the original CSV file
    input_csv_path = "BOOKS.csv"
    df = pd.read_csv(input_csv_path, error_bad_lines=False, sep=";")

    PRICES = []
    URLS = []
    for index, row in df.iterrows():
        book = row['TITLE']
        book = urllib.parse.quote_plus(book)

        LINK = "https://www.ebay.co.uk/sch/i.html?_from=R40&_trksid=p4432023.m570.l1313&_nkw={}&_sacat=0"
        LINK = LINK.format(book)
        print(LINK)

        prices, urls = get_prices_by_link(LINK)
        #print(len(urls))
        prices_without_outliers = remove_outliers(prices)
        #avg_price = get_average(prices_without_outliers)
        #print(avg_price)
        #save_to_file(avg_price)

        lws_price = get_lowest(prices_without_outliers)
        PRICES.append(lws_price)
        #print("Lowest Price:", lws_price[0])
        index = lws_price[1][0][0]
        url = urls[index]
        URLS.append(url)
        #print(url)

        #trained_model = train_price_prediction_model(prices_without_outliers)

        # Predict prices for the next 5 days
        #days_to_predict = 5
        #latest_price, future_prices = predict_price(trained_model, days_to_predict)

        #print("Latest Price:", latest_price)
        #print("Predicted Prices for the Next 5 Days:")
        #for i, price in enumerate(future_prices):
        #    print(f"Day {i+1}: {price:.2f}")



        # # Backup the data
        # #backup_data()

        # # Simulating data loss
        # prices = []
        # #restore_data()

        # # Restore the data
        # prices_restored = get_prices_by_link(LINK)
        # prices_without_outliers_restored = remove_outliers(prices_restored)
        # avg_price_restored = get_average(prices_without_outliers_restored)
        # print("Restored Data Average Price:", avg_price_restored)

    # Create a new column called 'URL' with the defined string
    df['URL'] = URLS
    df['PRICE'] = PRICES

    # Save the modified DataFrame to a new CSV file
    output_file = "BOOKS_with_URL.csv"
    df.to_csv(output_file, index=False)