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

PROJECTPATH = Path(os.getcwd())
DATAPATH = PROJECTPATH/'data/historical/gathered_data.parquet'

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


class DBUPDATE_PROCESSOR:
    def __init__(self):
        self.historical_df = pd.read_parquet(DATAPATH)
        self.outstanding_date = max(self.historical_df['Date'])
        logging.info(f'Outstanding date in DB is {self.outstanding_date.strftime("%Y-%m-%d")}')
        self.driver = self.start_driver()

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
    def update_db(self):
        logging.info('Searching race records to be updated...')
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
        logging.info(f'Number of date need to be updated: {len(date_list)}')
        for date in tqdm(list(filter(lambda x: x not in international_race,date_list)), leave=False):
            if datetime.datetime.now() < datetime.datetime.strptime(date,"%d/%m/%Y"):
                continue
            Select(WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select#selectId.f_fs11"))
            )).select_by_visible_text(date)
            self.driver.find_element(By.CSS_SELECTOR,"a#submitBtn").click()
            general_info = self.get_race_general_info()
            race = RACE(
                course=self.get_race_venue(date),
                date=datetime.datetime.strptime(date,"%d/%m/%Y"),
                meeting_number=self.get_meeting_number(),
                race_class=self.get_class(general_info),
                distance=self.get_distance(general_info),
                going=self.get_going(general_info),
                surface=self.get_surface(general_info),
                prize=self.get_prize(general_info),
                horse_race_detail=self.get_horse_detail()
            )

    def get_race_venue(self, date):
        venue = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "span.f_fl.f_fs13"))
        ).text
        return venue.split(date)[-1].strip()

    def get_race_general_info(self):
        return WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody.f_fs13"))).text

    def get_meeting_number(self):
        meeting_number = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "tr.bg_blue.color_w.font_wb"))
        ).text
        return int(re.search(r'RACE\s*(\d+)',meeting_number).group(1))

    def get_class(self, body):
        return re.search(r'(Class\s*\d+)', body).group(1).upper()

    def get_distance(self, body):
        return int(re.search(r'\s*(\d+)M\s*',body).group(1))

    def get_going(self, body):
        return re.search(r'Going\s*:\s*(\w+)\n',body).group(1)

    def get_surface(self, body):
        return re.search(r'Course\s*:\s*(.*?)\s*\n',body).group(1)

    def get_prize(self, body):
        return float(re.search(r'HK\$\s*(.*?)\s',body).group(1).replace(',',''))

    def get_horse_detail(self):
        horse_detail_table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.f_tac.table_bd.draggable"))
        )
        horse_detail_table = pd.read_html(horse_detail_table.get_attribute('outerHTML'))[0]
        horse_detail_table = horse_detail_table[horse_detail_table['Pla.'] != 'WV']
        return horse_detail_table


