import os
import re
from pathlib import Path
from dataclasses import dataclass, field
import datetime
import pandas as pd
import logging
import jockeyclub_betting.utils as utils
import time
import json
import numpy as np
import traceback
from tqdm import tqdm

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from data.historical.feature_col import cat_col
from ML.model.feature_col import index_col
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

PROJECTPATH = Path(os.getcwd())
#PROJECTPATH = Path(os.path.dirname(os.getcwd()))
DATAPATH = PROJECTPATH/'data/historical/'

@dataclass
class NEWHORSE:
    driver: any = field(init=False)
    input_df: pd.DataFrame
    horse_df: pd.DataFrame
    max_horse: int

    def __post_init__(self):
        self.driver = self.start_driver()

    @utils.logged
    def start_driver(self):
        logging.info('Starting chrome driver...')
        chrome_options = Options()
        chrome_options.page_load_strategy = "eager"
        chrome_options.add_argument('enable-automation')
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), chrome_options=chrome_options)
        #return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    def main(self):
        for horse in self.input_df['Horse_Code'].unique():
            horse_frame_new = self.update_horse_map(horse, self.max_horse + 1, self.input_df[self.input_df['Horse_Code'] == horse])
            horse_frame_new = horse_frame_new[['Horse_ID', 'Horse_Name', 'Horse_Code', 'Country', 'Import_type', 'Owner', 'Sire', 'Dam', 'Dam_sire']]
            self.horse_df = pd.concat([self.horse_df, horse_frame_new]).reset_index(drop=True)
            time.sleep(2)
        self.horse_df.to_parquet(DATAPATH / 'updated_horses.parquet')

    def update_horse_map(self, horsecode, horsenum, _df):
        self.driver.get('https://racing.hkjc.com/racing/information/English/Horse/SelectHorse.aspx')
        time.sleep(2)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="innerContent"]/div[2]/form[1]/table/tbody/tr[2]/td/table/tbody/tr/td/input[3]'))
        ).click()
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="innerContent"]/div[2]/form[1]/table/tbody/tr[2]/td/table/tbody/tr/td/input[4]'))
        ).send_keys(horsecode)
        time.sleep(1)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '// *[ @ id = "submit1"]'))
        ).click()
        second_column = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="innerContent"]/div[2]/div[1]/table[1]/tbody/tr/td[2]/table'))
        )
        second_column = pd.read_html(second_column.get_attribute('outerHTML'))[0]
        second_column.columns = ['second_key', 'comma', 'second_value']
        second_column.drop('comma', axis=1, inplace=True)
        second_column = second_column.set_index('second_key').T
        country = second_column[second_column.columns[second_column.columns.str.contains('Origin')]].values[0][0]
        country = country.split('/')[0]
        horse_id = horsenum
        import_type = second_column[second_column.columns[second_column.columns.str.contains('Import')]].values[0][0]
        third_column = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="innerContent"]/div[2]/div[1]/table[1]/tbody/tr/td[3]/table'))
        )
        third_column = pd.read_html(third_column.get_attribute('outerHTML'))[0]
        third_column.columns = ['third_key', 'comma', 'third_value']
        third_column.drop('comma', axis=1, inplace=True)
        third_column = third_column.set_index('third_key').T
        owner = third_column[third_column.columns[third_column.columns.str.contains('Owner')]].values[0][0]
        sire = third_column[third_column.columns[third_column.columns.str.fullmatch('Sire')]].values[0][0]
        dam = third_column[third_column.columns[third_column.columns.str.fullmatch('Dam')]].values[0][0]
        dam_sire = third_column[third_column.columns[third_column.columns.str.contains("Dam's")]].values[0][0]
        df = _df.copy()
        df['Country'] = country
        df['Import_type'] = import_type
        df['Horse_ID'] = horse_id
        df['Owner'] = owner
        df['Sire'] = sire
        df['Dam'] = dam
        df['Dam_sire'] = dam_sire
        return df

@dataclass
class SCRAPPER:
    driver: any = field(init=False)
    historical_df: pd.DataFrame = field(init=False)
    meeting_num: int


    def __post_init__(self):
        self.driver = self.start_driver()
        self.driver.get('https://racing.hkjc.com/racing/information/english/Racing/Racecard.aspx')
        if self.meeting_num > 1:
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(
                        #(By.XPATH, f'//*[@id="innerContent"]/div[2]/div[2]/table/tbody/tr/td[{self.meeting_num+1}]'))
                        (By.XPATH, f'//*[@id="innerContent"]/div[2]/div[3]/table/tbody/tr/td[{self.meeting_num+1}]/a'))
                ).click()
            except:
                pass
        self.historical_df = pd.read_parquet(DATAPATH/'historical_model_input.parquet')

    @utils.logged
    def start_driver(self):
        logging.info('Starting chrome driver...')
        chrome_options = Options()
        chrome_options.page_load_strategy = "eager"
        chrome_options.add_argument('enable-automation')
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), chrome_options=chrome_options)
        # return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    @utils.logged
    def get_race_info(self):
        try:
            meeting_info = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,'div.f_fs13'))
            ).text.split('\n')
            meeting_number = re.search(r'\d+',meeting_info[0]).group(0)
            logging.info(f'Processing meeting {meeting_number} general info...')
            date_venue = meeting_info[1].split(',')
            date = datetime.datetime.strptime(date_venue[1] + ' ' + date_venue[2], ' %B %d %Y')
            course = date_venue[3]
            race_id = self.historical_df['Race_ID'].max()+1
            rs_id = self.historical_df['RS_ID'].max()+1
            track_info = meeting_info[2].split(',')
            going = track_info[-1].upper()
            distance = int(track_info[-2].replace('M',''))
            surface = track_info[0].upper() + ' - ' + track_info[1].strip().replace('"','')
            prize_class = meeting_info[3].split(',')
            meeting_class = prize_class[-1].upper().strip()
            prize = float(re.search('\d+',prize_class[0]+prize_class[1]).group(0))
            number_of_turns = self.check_turns(course, surface, distance)
            sectional_id = self.historical_df['Sectional_ID'].max()+1
            logging.info(f'Processing meeting {meeting_number} horse info...')
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="innerContent"]/div[2]/div[5]/div[2]'))
            ).click()
            time.sleep(2)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '// *[ @ id = "ColSelectBody"] / form / table / tbody / tr[1] / td[4] / input'))
            ).click()
            time.sleep(2)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, '// *[ @ id = "ColSelectBody"] / form / table / thead / tr / td / table / tbody / tr / td[2] / a'))
            ).click()
            time.sleep(2)
            meeting_table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="racecardlist"]/tbody/tr/td/table'))
            )
            meeting_table = pd.read_html(meeting_table.get_attribute('outerHTML'))[0]
            meeting_table = meeting_table[meeting_table['Trainer'] != '-']
            runner = len(meeting_table)
            meeting_table = meeting_table[['Horse No.', 'Horse', 'Brand No.', 'Wt.', 'Jockey', 'Draw', 'Trainer', 'Horse Wt. (Declaration)']]
            meeting_table.columns = ['Horse_Number', 'Horse_Name', 'Horse_Code', 'Weight', 'Jockey', 'Draw', 'Trainer', 'Weight_Declared']
            meeting_table['Jockey'] = meeting_table['Jockey'].apply(lambda x: x.split('(')[0].strip())
            logging.info(f'Checking any new horse in meeting {meeting_number}...')
            horse_info = pd.read_parquet(DATAPATH/'updated_horses.parquet')
            find_new = meeting_table.merge(horse_info[['Horse_Code','Horse_Name']], how='left', indicator=True)
            new_horse = find_new[find_new['_merge'] == 'left_only'].drop('_merge', axis=1)
            new_horse.drop_duplicates(inplace=True)
            max_horse = horse_info['Horse_ID'].max()
            logging.info(f'New horse found: {len(new_horse)}')
            if len(new_horse) > 0:
                new_horse_processor = NEWHORSE(new_horse, horse_info, max_horse)
                new_horse_processor.main()
                horse_info = pd.read_parquet(DATAPATH / 'updated_horses.parquet')
            meeting_table = meeting_table.merge(horse_info[['Horse_ID','Horse_Code','Horse_Name','Country','Import_type','Owner','Sire','Dam','Dam_sire']],how='left',on=['Horse_Code','Horse_Name'])
            meeting_table['Horse_Unique_ID'] = meeting_table['Horse_Code'] + '_' + meeting_table['Horse_ID'].astype(int).astype(str)
            meeting_table['Date'] = date
            meeting_table['Race_ID'] = race_id
            meeting_table['RS_ID'] = rs_id
            meeting_table['Going'] = going
            meeting_table['Class'] = meeting_class
            meeting_table['Prize'] = prize
            meeting_table['Surface'] = surface
            meeting_table['Distance'] = distance
            meeting_table['number_of_turns'] = number_of_turns
            meeting_table['Sectional_ID'] = sectional_id
            meeting_table['Runners'] = runner
            meeting_table['Course'] = course
            meeting_table['Horse_ID'] = meeting_table['Horse_ID'].astype(str)
            #meeting_table['Place'] = meeting_table['Place'].astype(str)
            meeting_table['Weight_Declared'] = meeting_table['Weight_Declared'].astype(int)
            historical_dtype = {col : self.historical_df[col].dtype for col in self.historical_df.columns}
            self.historical_df = pd.concat([self.historical_df, meeting_table.drop(['Horse_Name'], axis=1)])
            self.calculate_feature()
            try:
                for key, val in historical_dtype.items():
                    if key not in index_col and key != 'Ground_Truth' and key != 'Place' and key not in cat_col:
                        self.historical_df[key] = self.historical_df[key].astype(str).astype(val)
            except:
                logging.info(traceback.format_exc())
            meeting_table = self.historical_df[self.historical_df['Race_ID'] == race_id]
            return meeting_table
        except:
            logging.info(traceback.format_exc())

    def check_turns(self,course,suface,distance):
        if course == 'Sha Tin':
            if 'TURF' in suface:
                if distance == 1000:
                    return 0
                elif distance > 1000 and distance <= 1800:
                    return 1
                else:
                    return 2
            else:
                if distance == 1200:
                    return 1
                elif distance > 1200 and distance <= 2000:
                    return 2
                else:
                    return 3
        else:
            if distance == 1000:
                return 1.5
            elif distance > 1000 and distance <= 1200:
                return 2.5
            elif distance >1200 and distance <= 1800:
                return 3.5
            else:
                return 5

    def calculate_feature(self):
        old_df = self.historical_df[self.historical_df['Date'] <= datetime.datetime(2022,12,18)]
        old_jockey = old_df[['Jockey','Jockey_ID']].drop_duplicates()
        old_jockey = old_jockey.to_json(orient='values')
        old_jockey = {pair[0]:pair[1] for pair in json.loads(old_jockey)}
        old_trainer = old_df[['Trainer','Trainer_ID']].drop_duplicates()
        old_trainer = old_trainer.to_json(orient='values')
        old_trainer = {pair[0]:pair[1] for pair in json.loads(old_trainer)}
        self.historical_df['Jockey_ID'] = self.historical_df['Jockey'].apply(lambda x: old_jockey.setdefault(x, len(old_jockey) + 1)).astype(int)
        self.historical_df['Trainer_ID'] = self.historical_df['Trainer'].apply(lambda x: old_trainer.setdefault(x, len(old_trainer)+1)).astype(int)
        #self.historical_df.drop(['Win_odds','Horse_Name'], axis=1, inplace=True)
        self.historical_df['Distance'] = self.historical_df['Distance'].astype(int)
        self.historical_df['Draw'] = self.historical_df['Draw'].astype(int)
        self.historical_df.sort_values(['Horse_Unique_ID', 'Date'], inplace=True)
        self.historical_df['Place'] = self.historical_df['Place'].astype(str).apply(lambda x: re.sub('\D*','',x))
        self.historical_df['Place'] = self.historical_df['Place'].replace(r'', '0')
        #self.historical_df = self.historical_df[self.historical_df['Place'] != '']
        self.historical_df['Place'] = self.historical_df['Place'].astype(int)
        self.historical_df['prev_place'] = self.historical_df.groupby('Horse_Unique_ID')['Place'].shift()
        self.historical_df['prev_place'] = (self.historical_df['prev_place'] <= 3)
        self.historical_df['Weight'] = self.historical_df['Weight'].astype(int)
        self.historical_df['prev_weight'] = self.historical_df.groupby('Horse_Unique_ID')['Weight'].shift()
        self.historical_df['prev_weight_diff'] = self.historical_df['Weight'] - self.historical_df['prev_weight']
        self.historical_df['prev_place'] = self.historical_df['prev_place'].fillna(False)
        self.historical_df['prev_weight_diff'] = self.historical_df['prev_weight_diff'].fillna(0)
        self.historical_df['prev_place'] = self.historical_df['prev_place'] * 1
        self.historical_df['prev_place'] = self.historical_df['prev_place'].fillna(0)
        self.historical_df['winning_weight'] = self.historical_df[['prev_weight', 'prev_place']].apply(lambda x: x['prev_weight'] if x['prev_place'] else 0, axis=1)
        self.historical_df['cumsum_weight'] = self.historical_df.groupby('Horse_Unique_ID')['winning_weight'].cumsum()
        self.historical_df['cumcount_win'] = self.historical_df.groupby(['Horse_Unique_ID'])['prev_place'].cumsum()
        self.historical_df['avg_winning_weight'] = self.historical_df[['cumsum_weight', 'cumcount_win']].apply(lambda x: (x['cumsum_weight'] / x['cumcount_win']) if x['cumcount_win'] != 0 else 0, axis=1)
        self.historical_df['wtdf_prev_avg_win_wt'] = self.historical_df[['avg_winning_weight', 'Weight']].apply(lambda x: 0 if x['avg_winning_weight'] == 0 else x['Weight'] - x['avg_winning_weight'], axis=1)
        self.historical_df.drop(['prev_weight', 'avg_winning_weight', 'cumsum_weight', 'winning_weight'], axis=1, inplace=True)
        self.historical_df['prev_raced'] = self.historical_df.groupby(['Horse_Unique_ID'])['Place'].shift(1).fillna(0)
        self.historical_df['prev_raced'] = self.historical_df['prev_raced'].apply(lambda x: 1 if x != 0 else 0)
        self.historical_df['prev_draw1'] = self.historical_df.groupby(['Horse_Unique_ID'])['Draw'].shift(1).fillna(0)
        self.historical_df['prev_draw2'] = self.historical_df.groupby(['Horse_Unique_ID'])['Draw'].shift(2).fillna(0)
        self.historical_df['prev_Jockey1'] = self.historical_df.groupby(['Horse_Unique_ID'])['Jockey'].shift(1)
        self.historical_df['prev_Jockey2'] = self.historical_df.groupby(['Horse_Unique_ID'])['Jockey'].shift(2)
        self.historical_df['prev_class'] = self.historical_df.groupby(['Horse_Unique_ID'])['Class'].shift(1)
        self.historical_df['prev_class'] = self.historical_df[['Class', 'prev_place']].apply(lambda x: x['Class'] if x is None else x['prev_place'],axis=1)
        self.historical_df['prev_dist'] = self.historical_df.groupby(['Horse_Unique_ID'])['Distance'].shift(1)
        self.historical_df.sort_values(['Horse_Unique_ID', 'Date'], inplace=True, ascending=[True, False])
        df2 = self.historical_df.copy()
        df2['horse_acc_q'] = df2['Place'].apply(lambda x: 1 if x <= 3 else 0)
        df2['horse_acc_win'] = df2['Place'].apply(lambda x: 1 if x == 1 else 0)
        cum_win_q = df2[['Horse_Unique_ID', 'Date', 'horse_acc_win', 'horse_acc_q']].groupby(['Horse_Unique_ID', 'Date']).sum().groupby(level=0).cumsum()
        sum_win_q = df2[['Horse_Unique_ID', 'Date', 'horse_acc_win', 'horse_acc_q']].groupby(['Horse_Unique_ID', 'Date']).sum()
        sum_win_q.rename({'horse_acc_win': 'win_on_that_day', 'horse_acc_q': 'q_on_that_day'}, axis=1, inplace=True)
        output = pd.merge(cum_win_q, sum_win_q, left_index=True, right_index=True)
        output['horse_winning_history'] = output['horse_acc_win'].shift(periods=1, fill_value=0, axis=0)
        output['horse_qing_history'] = output['horse_acc_q'].shift(periods=1, fill_value=0, axis=0)
        output['horse_last_5_day_q'] = output['q_on_that_day'].rolling(5, closed='left').mean()
        output['horse_last_5_day_win'] = output['win_on_that_day'].rolling(5, closed='left').mean()
        output['horse_last_2_day_q'] = output['q_on_that_day'].rolling(2, closed='left').mean()
        output['horse_last_2_day_win'] = output['win_on_that_day'].rolling(2, closed='left').mean()
        output['horse_last5_vs_2q'] = output['horse_last_5_day_q'] - output['horse_last_2_day_q']
        output['horse_last5_vs_2w'] = output['horse_last_5_day_win'] - output['horse_last_2_day_win']
        output.drop(['horse_acc_win', 'horse_acc_q', 'win_on_that_day', 'q_on_that_day'], axis=1, inplace=True)
        self.historical_df = self.historical_df.drop(['horse_winning_history', 'horse_qing_history', 'horse_last_5_day_q','horse_last_5_day_win','horse_last_2_day_q','horse_last_2_day_win','horse_last5_vs_2q','horse_last5_vs_2w'],axis=1)\
            .merge(output, how='inner', on=['Horse_Unique_ID', 'Date'])
        self.historical_df.sort_values(['Horse_Unique_ID', 'Date'], inplace=True)
        del output, df2
        df2 = self.historical_df.copy()
        df2['jockey_acc_q'] = df2['Place'].apply(lambda x: 1 if x <= 3 else 0)
        df2['jockey_acc_win'] = df2['Place'].apply(lambda x: 1 if x == 1 else 0)
        cum_win_q = df2[['Jockey_ID', 'Date', 'jockey_acc_win', 'jockey_acc_q']].groupby(['Jockey_ID', 'Date']).sum().groupby(level=0).cumsum()
        sum_win_q = df2[['Jockey_ID', 'Date', 'jockey_acc_win', 'jockey_acc_q']].groupby(['Jockey_ID', 'Date']).sum()
        sum_win_q.rename({'jockey_acc_win': 'win_on_that_day', 'jockey_acc_q': 'q_on_that_day'}, axis=1, inplace=True)
        output = pd.merge(cum_win_q, sum_win_q, left_index=True, right_index=True)
        output['jockey_winning_history'] = output['jockey_acc_win'].shift(periods=1, fill_value=0, axis=0)
        output['jockey_qing_history'] = output['jockey_acc_q'].shift(periods=1, fill_value=0, axis=0)
        output['last_200_day_q'] = output['q_on_that_day'].rolling(200, closed='left').mean()
        output['last_200_day_win'] = output['win_on_that_day'].rolling(200, closed='left').mean()
        output['last_500_day_q'] = output['q_on_that_day'].rolling(500, closed='left').mean()
        output['last_500_day_win'] = output['win_on_that_day'].rolling(500, closed='left').mean()
        output['jockey_last500_vs_200_w'] = output['last_500_day_win'] - output['last_200_day_win']
        output['jockey_last500_vs_200_q'] = output['last_500_day_q'] - output['last_200_day_q']
        output.drop(['jockey_acc_win', 'jockey_acc_q', 'win_on_that_day', 'q_on_that_day'], axis=1, inplace=True)
        self.historical_df = self.historical_df.drop(['jockey_winning_history','jockey_qing_history','last_200_day_q','last_200_day_win','last_500_day_q','last_500_day_win','jockey_last500_vs_200_w','jockey_last500_vs_200_q'],axis=1)\
            .merge(output, how='inner', on=['Jockey_ID', 'Date'])
        del output, df2
        self.historical_df['horse2_vs_jockey200'] = self.historical_df['last_200_day_q'] * self.historical_df['horse_last_2_day_q']
        self.historical_df['horse2_vs_jockey200_win'] = self.historical_df['last_200_day_q'] * self.historical_df['horse_last_2_day_win']
        self.historical_df['horse5_vs_jockey200_win'] = self.historical_df['last_200_day_q'] * self.historical_df['horse_last_5_day_win']
        self.historical_df['horse5_vs_jockey200'] = self.historical_df['last_200_day_q'] * self.historical_df['horse_last_5_day_q']
        self.historical_df['j_draw'] = self.historical_df['Jockey'].astype(str) + '_' + self.historical_df['Draw'].astype(str)
        self.historical_df['j_draw'] = self.historical_df['j_draw'].astype("category")
        self.historical_df = self.historical_df[self.historical_df['Weight_Declared'] != '---']
        self.historical_df['Weight'] = self.historical_df['Weight'].astype(int)
        self.historical_df['Weight_Declared'] = self.historical_df['Weight_Declared'].astype(int)
        self.historical_df.sort_values(['Date','Race_ID'], inplace=True)

def upload_to_ggldrive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")
    drive = GoogleDrive(gauth)
    upload_file_list = ['prediction_meeting_overall.csv']
    file_path = PROJECTPATH/'prediction/'
    for upload_file in upload_file_list:
        gfile = drive.CreateFile({'parents': [{'id': ''}], 'title': upload_file})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(os.path.join(file_path,upload_file))
        gfile.Upload() # Upload the file.

@utils.logged
def main():
    def predict(model, df):
        return model.predict(df.loc[:, ~df.columns.isin(index_col)])

    chrome_options = Options()
    chrome_options.page_load_strategy = "eager"
    chrome_options.add_argument('enable-automation')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), chrome_options=chrome_options)
    #driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get('https://racing.hkjc.com/racing/information/english/Racing/Racecard.aspx')
    meeting_bar = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.f_fs12.js_racecard"))
    )
    meeting_bar = pd.read_html(meeting_bar.get_attribute('outerHTML'))[0]
    total_meeting = len(meeting_bar.columns)
    logging.info('Loading model...')
    xgb_model = xgb.XGBRanker()
    xgb_model.load_model(PROJECTPATH/'ML/model/xgboost_ranker.json')
    #meeting = 1
    overall = []
    for meeting in tqdm(range(1, total_meeting), leave=False):
        logging.info(f'Predicting meeting number: {meeting} .')
        scrapper = SCRAPPER(meeting)
        try:
            df = scrapper.get_race_info()
            origin = df.copy()
            df = df[["Course","Class","Distance","Going","Surface","Runners","Prize","number_of_turns","Weight","Weight_Declared","Draw","Jockey","Trainer","Country","Import_type","Owner","Sire","Dam","Dam_sire","prev_place","prev_weight_diff","cumcount_win","wtdf_prev_avg_win_wt","prev_raced","prev_class","prev_dist","horse_winning_history","horse_qing_history","horse_last_5_day_q","horse_last_5_day_win","horse_last_2_day_q","horse_last_2_day_win","horse_last5_vs_2q","horse_last5_vs_2w","jockey_winning_history","jockey_qing_history","last_200_day_q","last_200_day_win","last_500_day_q","last_500_day_win","jockey_last500_vs_200_w","jockey_last500_vs_200_q","horse2_vs_jockey200","horse2_vs_jockey200_win","horse5_vs_jockey200_win","horse5_vs_jockey200"]+['Race_ID']]
            for col in cat_col:
                df[col] = df[col].astype('category')
            prediction = df.groupby('Race_ID').apply(lambda x: predict(xgb_model, x))
        except:
            logging.info(traceback.format_exc())
            print(f'Meeting number {meeting} failed to scrap...')
            continue
            #raise Exception('model error...')
        prediction = pd.DataFrame(prediction).reset_index()
        prediction.columns = ['Race_ID', 'predict_result']
        results = prediction['predict_result'].apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 'value']].set_index('index')
        final_prediction = pd.merge(prediction[['Race_ID']], results, left_index=True, right_index=True)
        final_prediction['rank'] = final_prediction.groupby("Race_ID")["value"].rank(method="dense", ascending=False)
        final_prediction = final_prediction.reset_index()
        final_prediction = pd.concat([origin.reset_index(drop=True),final_prediction[['rank']]],axis=1)
        final_prediction = final_prediction[['Date','Race_ID','Horse_Number','Draw','Horse_Unique_ID','rank']].sort_values('rank')
        final_prediction['Meeting'] = meeting
        final_prediction.to_csv(PROJECTPATH/f'prediction/prediction_meeting_{meeting}.csv',index=False)
        overall.append(final_prediction)
    overall = pd.concat(overall)
    overall.to_csv(PROJECTPATH/f'prediction/prediction_meeting_overall.csv',index=False)
    upload_to_ggldrive()
if __name__ == '__main__':
    main()

