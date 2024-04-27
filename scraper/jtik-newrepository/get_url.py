import time 
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By


driver = webdriver.Firefox() 

target = [
    {
        "url": "http://jurusan.tik.pnj.ac.id/newrepositori/v2/kategori/1",
        "max_page": 36,
        "prodi": "TI"
    },
    {
        "url": "http://jurusan.tik.pnj.ac.id/newrepositori/v2/kategori/2",
        "max_page": 12,
        "prodi": "TMJ"
    },
    {
        "url": "http://jurusan.tik.pnj.ac.id/newrepositori/v2/kategori/3",
        "max_page": 6,
        "prodi": "TMD"
    },
]

url = []
prodi = []

for obj in target:
    for page in range(1, obj["max_page"] + 1):
        driver.get(f"{obj['url']}?page={page}")
        time.sleep(1) 

        div = driver.find_elements(By.CLASS_NAME, "col-md-10")

        for element in div:
            link_element = element.find_element(By.TAG_NAME, "a")
            href_value = link_element.get_attribute("href")
            url.append(href_value)
            prodi.append(obj["prodi"])

df = pd.DataFrame({"url": url, "prodi": prodi})
df.to_csv('url.csv', index=False) 

driver.quit()