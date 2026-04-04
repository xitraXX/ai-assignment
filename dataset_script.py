import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

url = "https://www.tarc.edu.my/admissions/programmes/programme-offered-a-z/undergraduate-programme/" # undergraduate programmes
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Example: Scraping a list of program names
    # Update these selectors based on your 'Inspect Element' results
    programs = soup.find_all('div', class_='tab-content text-secondary')

    data = []
    for p in programs:
        name = p.text.strip()

    # Simple check to skip the alphabet bar A|B|C
        if "|" in name or len(name) <= 1:
            category = "none"
        
        # This creates a SEPARATE row for every 'p' in 'programs'
        data.append({
            "User_Message": name,
            "Response": f"Yes, we offer that {category}, the program name is: {name}.",
            "Intent": f"{category}_{name.replace(' ', '_').lower()}"
        })

    df = pd.DataFrame(data)
    file_name = 'tarumt_dataset.csv'

    df.to_csv(file_name, mode='a', index=False, header=not os.path.exists(file_name))
    print(f"Scraped {len(data)} items into {file_name}")

    # Load the final dataset (after scrap, clean duplicate dataset)
    df = pd.read_csv('tarumt_dataset.csv')

    # Drop exact duplicate rows and save it back
    df.drop_duplicates(inplace=True)
    df.to_csv('tarumt_dataset_cleaned.csv', index=False)
