import requests
import shutil
import os
import pandas as pd
import urllib

BOOK_API_URL = "https://www.googleapis.com/books/v1/volumes?q="

def get_isbn(title):
    try:
        query = f"{title}"
        response = requests.get(BOOK_API_URL + query)
        json_response = response.json()
        volume_info = json_response['items'][0]['volumeInfo']
        return volume_info.get('industryIdentifiers')[0].get('identifier')
    except:
        return 'NULL'
if __name__ == '__main__':

    # Read the original CSV file
    input_csv_path = "BOOKS.csv"
    df = pd.read_csv(input_csv_path, error_bad_lines=False, sep=",", encoding = "utf-8")

    ISBNS = []

    for index, row in df.iterrows():
        book = row['TITLE']
        book = urllib.parse.quote_plus(book)

        ISBN = get_isbn(book)
        print(ISBN)

        ISBNS.append(ISBN)



    # Create a new column called 'URL' with the defined string
    df['ISBN'] = ISBNS

    # Save the modified DataFrame to a new CSV file
    output_file = "BOOKS_with_ISBN.csv"
    df.to_csv(output_file, index=False)