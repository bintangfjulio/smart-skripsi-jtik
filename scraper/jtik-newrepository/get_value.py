import time 
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By


driver = webdriver.Firefox() 
df = pd.read_csv('url.csv')

url = []
prodi = []
judul = []
abstrak = []
kata_kunci = []

for _, row in df.iterrows():
    driver.get(row['url'])
    time.sleep(1)

    div_1 = driver.find_element(By.CLASS_NAME, "col-md-12.content-journal-doc-page")
    div_2 = driver.find_element(By.CLASS_NAME, "col-md-4")
    
    data_judul = div_1.find_element(By.CLASS_NAME, "font-weight-bold").text
    data_abstrak = div_1.find_element(By.TAG_NAME, "p").text
    data_kata_kunci = div_2.find_element(By.TAG_NAME, "span").text

    url.append(row['url'])
    prodi.append(row['prodi'])
    judul.append(data_judul)
    abstrak.append(data_abstrak)
    kata_kunci.append(data_kata_kunci)

df = pd.DataFrame({"url": url, "prodi": prodi, "kata_kunci": kata_kunci, "judul": judul, "abstrak": abstrak})
df.to_csv("data_newrepo_jtik.csv", index=False)

driver.quit()