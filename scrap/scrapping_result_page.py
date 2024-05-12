import pandas as pd
from pathlib import Path
import numpy as np
import os
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import jockeyclub_betting.utils as utils
import logging
from scrap.ext import international_race
from dataclasses import dataclass, field
import time
import datetime
import re
from tqdm import tqdm
from selenium.webdriver.chrome.options import Options

PROJECTPATH = Path(os.getcwd())
DATAPATH = PROJECTPATH/'data/historical/gathered_data.parquet'

historical_df = pd.read_parquet(DATAPATH)
outstanding_date = max(historical_df['Date'])

def start_driver():
    logging.info('Starting chrome driver...')
    return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

driver = start_driver()

def head_to_result_pg():
    url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx'
    logging.info('Opening race result page in HKJC...')
    driver.get(url)

head_to_result_pg()

date_list = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "select#selectId.f_fs11"))
).text.split('\n')
date_list = list(map(lambda x: x.strip(), date_list))

for i, x in enumerate(date_list):
    if x == outstanding_date.strftime("%d/%m/%Y"):
        if i == 0:
            break
        date_list = date_list[:i]
        break

def get_race_general_info():
    return WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody.f_fs13"))).text


date_list.reverse()
date = date_list[0]
Select(WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "select#selectId.f_fs11"))
)).select_by_visible_text(date)
driver.find_element(By.CSS_SELECTOR,"a#submitBtn").click()
general_info = get_race_general_info()


def get_race_venue(date):
    venue = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "span.f_fl.f_fs13"))
    ).text
    return venue.split(date)[-1].strip()

def get_meeting_number():
    meeting_number = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "tr.bg_blue.color_w.font_wb"))
    ).text
    return int(re.search(r'RACE\s*(\d+)', meeting_number).group(1))

def get_class(body):
    return re.search(r'(Class\s*\d+)', body).group(1).upper()

def get_distance(body):
    return int(re.search(r'\s*(\d+)M\s*', body).group(1))

def get_surface(body):
    return re.search(r'Course\s*:\s*(.*?)\s*\n', body).group(1)

def get_prize(body):
    return float(re.search(r'HK\$\s*(.*?)\s', body).group(1).replace(',', ''))

def get_horse_detail():
    horse_detail_table = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.f_tac.table_bd.draggable"))
    )
    horse_detail_table = pd.read_html(horse_detail_table.get_attribute('outerHTML'))[0]
    horse_detail_table = horse_detail_table[horse_detail_table['Pla.'] != 'WV']
    return horse_detail_table

def get_going(body):
    return re.search(r'Going\s*:\s*(.*?)\n', body).group(1)

@dataclass
class RACE:
    course: str
    date: datetime.datetime
    meeting_number: int
    race_class: str
    distance: int
    going: str
    surface: str
    prize: float
    horse_race_detail: pd.DataFrame

race = RACE(
    course=get_race_venue(date),
    date=datetime.datetime.strptime(date, "%d/%m/%Y"),
    meeting_number=get_meeting_number(),
    race_class=get_class(general_info),
    distance=get_distance(general_info),
    going=get_going(general_info),
    surface=get_surface(general_info),
    prize=get_prize(general_info),
    horse_race_detail=get_horse_detail()
)


