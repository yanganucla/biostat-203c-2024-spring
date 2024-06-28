import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


def parse_full_credits(movie_directory):
    # Load movie's full credits page content
    url = f'https://www.themoviedb.org/movie/{movie_directory}/cast'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find the cast section
    cast_section = soup.find('ol', class_='people credits')
    actor_set = set()
    df = pd.DataFrame(columns=['actor', 'movie_or_TV_name'])
    for row in cast_section.select('div.info a'):
        actor_directory = row['href'].replace('/person/', '')
        if actor_directory not in actor_set:
            df = parse_actor_page(df, actor_directory)
            print(actor_directory)
            actor_set.add(actor_directory)
            time.sleep(0.5)
    # Sort the DataFrame by actor then movie_or_TV_name
    df_sorted = df.sort_values(by=['actor', 'movie_or_TV_name'])
    return df_sorted


def parse_actor_page(df, actor_directory):
    # Load actor's page content
    url = f'https://www.themoviedb.org/person/{actor_directory}'
    response = requests.get(url=url)
    if response.status_code != 200:
        return df
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the actor's name
    actor_name = soup.find('head').find('title').get_text().split(' — ')[0]

    # Find the section with "Acting"
    acting_section = None
    for h3 in soup.select_one('section.credits').find_all('h3'):
        if 'Acting' in h3.get_text():
            acting_section = h3
            break
    if acting_section:
        # The next sibling of the "Acting" header is the table
        acting_table = acting_section.find_next_sibling('table')
        if acting_table:
            title_set = set()
            for row in acting_table.select('a.tooltip'):
                title = row.get_text(strip=True)
                if title not in title_set:
                    new_row = pd.DataFrame({'actor': [actor_name], 'movie_or_TV_name': [title]})
                    df = pd.concat([df, new_row], ignore_index=True)
                    title_set.add(title)
    return df


if __name__ == '__main__':
    # df = pd.DataFrame(columns=['actor', 'movie_or_TV_name'])
    # df = parse_actor_page(df, "2710-james-cameron")
    parse_full_credits('671-harry-potter-and-the-philosopher-s-stone')