import pandas as pd
from pathlib import Path
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
import time
import datetime

PROJECTPATH = Path(os.getcwd())
DATAPATH = PROJECTPATH/'data/historical/gathered_data.parquet'

class DBMANAGER:
    def __init__(self):
        self.historical_df = pd.read_parquet(DATAPATH)
        self.outstanding_date = max(self.historical_df['Date'])
        logging.info(f'Outstanding date in DB is {self.outstanding_date.strftime("%Y-%m-%d")}')
        self.driver = self.start_driver()
        self.venue = None

    @utils.logged
    def start_driver(self):
        logging.info('Starting chrome driver...')
        return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    @utils.logged
    def head_to_result_pg(self):
        url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx'
        logging.info('Opening race result page in HKJC...')
        self.driver.get(url)

    @utils.logged
    def select_date(self):
        date_list = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select#selectId.f_fs11"))
        ).text.split('\n')
        date_list = list(map(lambda x: x.strip(), date_list))
        date_list = [i for i in date_list if i != '']
        for i, x in enumerate(date_list):
            if x == self.outstanding_date.strftime("%d/%m/%Y"):
                if i == 0:
                    return None
                date_list = date_list[:i]
                break
        date_list.reverse()
        for date in list(filter(lambda x: x not in international_race,date_list)):
            if datetime.datetime.now() < datetime.datetime.strptime(date,"%d/%m/%Y"):
                continue
            Select(WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select#selectId.f_fs11"))
            )).select_by_visible_text(date)
            self.driver.find_element(By.CSS_SELECTOR,"a#submitBtn").click()
            self.get_race_venue(date)

    def get_race_venue(self, date):
        self.venue = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "span.f_fl.f_fs13"))
        ).text
        self.venue = self.venue.split(date.strftime("%d/%m/%Y"))[-1].strip()

