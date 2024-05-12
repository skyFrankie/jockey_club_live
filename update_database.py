import pandas as pd
from pathlib import Path
#from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import os
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
import jockeyclub_betting.utils as utils
import logging
from scrap.ext import international_race
from dataclasses import dataclass, field
import time
import datetime
import re
from tqdm import tqdm
import json
from data.historical.feature_col import cat_col
import traceback
import warnings
warnings.filterwarnings('ignore')

PROJECTPATH = Path(os.path.dirname(os.getcwd()))
#PROJECTPATH = Path(os.getcwd())
DATAPATH = PROJECTPATH/'data/historical/'

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
    race_id: int
    rs_id: int
    sectional_id: int
    record_id: int
    driver: any = field(init=False)

    def __post_init__(self):
        self.horse_race_detail['Course'] = self.course
        self.horse_race_detail['Class'] = self.race_class
        self.horse_race_detail['Distance'] = int(self.distance)
        self.horse_race_detail['Going'] = self.going
        self.horse_race_detail['Surface'] = self.surface.replace('"','')
        self.horse_race_detail['Date'] = self.date
        self.horse_race_detail['Prize'] = float(self.prize)
        self.horse_race_detail['Race_ID'] = self.race_id
        self.prize_inflation()
        self.horse_race_detail['number_of_turns'] = self.horse_race_detail[['Course', 'Surface', 'Distance']].apply(lambda x: self.check_turns(x['Course'], x['Surface'], x['Distance']), axis=1)
        #self.get_course_cluster()
        self.horse_race_detail['RS_ID'] = self.rs_id
        self.horse_race_detail['Runners'] = len(self.horse_race_detail)
        self.horse_race_detail['Sectional_ID'] = self.sectional_id
        self.horse_race_detail.reset_index(inplace=True)
        self.horse_race_detail.rename({'index':'Record_ID'},axis=1,inplace=True)
        self.horse_race_detail['Record_ID'] = self.horse_race_detail['Record_ID']+1+self.record_id
        self.map_historical_horse()
        self.horse_race_detail['Horse_Unique_ID'] = self.horse_race_detail['Horse_Code'] + '_' + self.horse_race_detail['Horse_ID'].astype(int).astype(str)
        cond_list = [self.horse_race_detail['Place'] == 1, self.horse_race_detail['Place'] == 2, self.horse_race_detail['Place'] == 3, self.horse_race_detail['Place'] == 4, self.horse_race_detail['Place'] == 5,
                     self.horse_race_detail['Place'] == 6, self.horse_race_detail['Place'] == 7, self.horse_race_detail['Place'] == 8, self.horse_race_detail['Place'] == 9, self.horse_race_detail['Place'] == 10,
                     self.horse_race_detail['Place'] == 11, self.horse_race_detail['Place'] == 12, self.horse_race_detail['Place'] == 13, self.horse_race_detail['Place'] == 14]
        choice_list = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        self.horse_race_detail['Ground_Truth'] = np.select(cond_list, choice_list, 0)

    def prize_inflation(self):
        current_year = datetime.date.today().year
        self.horse_race_detail['year'] = self.horse_race_detail['Date'].apply(lambda x: current_year - x.year)
        self.horse_race_detail['Prize'] = self.horse_race_detail[['Prize', 'year']].apply(lambda x: x['Prize'] * (1.025 ** x['year']) if x['year'] != 0 else x['Prize'], axis=1)
        self.horse_race_detail.drop('year', axis=1, inplace=True)
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

    # def get_course_cluster(self):
    #     X = self.horse_race_detail[['Race_ID', 'Course', 'Distance', 'Going', 'Surface', 'Class', 'number_of_turns']]
    #     X = pd.get_dummies(X.drop('Race_ID', axis=1))
    #     Z = linkage(X, 'ward')
    #     clusters = fcluster(Z, 10, criterion='maxclust')
    #     self.horse_race_detail['Course_Cluster'] = clusters

    def map_historical_horse(self):
        chrome_options = Options()
        chrome_options.page_load_strategy = "eager"
        chrome_options.add_argument('enable-automation')
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        #check updated horses parquet exists
        if not os.path.exists(DATAPATH/'updated_horses.parquet'):
            temp = pd.read_json(DATAPATH/'horseracedatabase_horses.json')
        else:
            temp = pd.read_parquet(DATAPATH/'updated_horses.parquet')
        find_new = self.horse_race_detail.merge(temp[['Horse_Code','Horse_Name']],how='left',indicator=True)
        new_horse = find_new[find_new['_merge'] == 'left_only'].drop('_merge',axis=1)
        #new_horse = self.horse_race_detail[(~self.horse_race_detail['Horse_Name'].isin(temp['Horse_Name']))&(~self.horse_race_detail['Horse_Code'].isin(temp['Horse_Code']))]
        #new_horse = new_horse[~new_horse['Horse_Code'].isin(temp['Horse_Code'])]
        new_horse.drop_duplicates(inplace=True)
        max_horse = temp['Horse_ID'].max()
        if len(new_horse) > 0:
            self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), chrome_options=chrome_options)
            for horse in tqdm(new_horse['Horse_Code'].unique(), leave=False):
                horse_frame_new = self.update_horse_map(horse, max_horse+1, new_horse[new_horse['Horse_Code'] == horse])
                horse_frame_new = horse_frame_new[['Horse_ID','Horse_Name','Horse_Code','Country','Import_type','Owner','Sire','Dam','Dam_sire']]
                temp = pd.concat([temp, horse_frame_new]).reset_index(drop=True)
                time.sleep(2)
            temp.to_parquet(DATAPATH/'updated_horses.parquet')
        self.horse_race_detail = self.horse_race_detail.merge(temp[['Horse_Name','Horse_Code','Horse_ID','Country','Import_type','Owner','Sire','Dam','Dam_sire']],how='left',on=['Horse_Name','Horse_Code'])
        # new_flag = self.horse_race_detail['Horse_ID'].isnull()
        # if new_flag.sum() > 0:
        #     self.horse_race_detail.loc[new_flag, 'Horse_ID'] = range(temp['Horse_ID'].max()+1, temp['Horse_ID'].max()+1+new_flag.sum())
        #     self.horse_race_detail['Horse_ID'] = self.horse_race_detail['Horse_ID'].astype(str)

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


class DBUPDATE_PROCESSOR:
    def __init__(self):
        self.historical_df = pd.read_parquet(DATAPATH/'historical_model_input.parquet')
        self.outstanding_date = max(self.historical_df['Date'])
        self.url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx'
        logging.info(f'Outstanding date in DB is {self.outstanding_date.strftime("%Y-%m-%d")}')
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

    @utils.logged
    def head_to_result_pg(self):
        logging.info('Opening race result page in HKJC...')
        self.driver.get(self.url)
        time.sleep(2)

    @utils.logged
    def update_db(self):
        time.sleep(2)
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
        #filter date that is international_race
        for date in tqdm(list(filter(lambda x: x not in international_race,date_list)), leave=False):
            if datetime.date.today() <= datetime.datetime.strptime(date,"%d/%m/%Y").date():
                break
            #select date
            Select(WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select#selectId.f_fs11"))
            )).select_by_visible_text(date)
            #submit date
            self.driver.find_element(By.CSS_SELECTOR,"a#submitBtn").click()
            #check number of meeting
            meeting_table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.f_fs12.js_racecard"))
            )
            meeting_table = pd.read_html(meeting_table.get_attribute('outerHTML'))[0]
            meeting_number = len(meeting_table.columns)
            for meeting in tqdm(range(1,meeting_number-1),leave=False):
                if meeting != 1:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(
                            (By.XPATH, f'//*[@id="innerContent"]/div[2]/div[2]/table/tbody/tr/td[{meeting+1}]'))
                    ).click()
                time.sleep(2)
                try:
                    general_info = self.get_race_general_info()
                except Exception as e:
                    self.driver.get(self.url)
                    time.sleep(2)
                    Select(WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "select#selectId.f_fs11"))
                    )).select_by_visible_text(date)
                    self.driver.find_element(By.CSS_SELECTOR, "a#submitBtn").click()
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(
                            (By.XPATH, f'//*[@id="innerContent"]/div[2]/div[2]/table/tbody/tr/td[{meeting+1}]'))
                    ).click()
                    time.sleep(2)
                    general_info = self.get_race_general_info()
                try:
                    race = RACE(
                        course=self.get_race_venue(date),
                        date=datetime.datetime.strptime(date,"%d/%m/%Y"),
                        meeting_number=self.get_meeting_number(),
                        race_class=self.get_class(general_info),
                        distance=self.get_distance(general_info),
                        going=self.get_going(general_info),
                        surface=self.get_surface(general_info),
                        prize=self.get_prize(general_info),
                        horse_race_detail=self.get_horse_detail(),
                        race_id=self.get_race_id(),
                        rs_id=self.get_rs_id(),
                        sectional_id=self.get_sectional_id(),
                        record_id=self.historical_df['Record_ID'].max()
                    )
                    self.historical_df = pd.concat([self.historical_df,race.horse_race_detail.drop(['Horse_Name'],axis=1)])
                except Exception as e:
                    logging.info(e)
                    pass
            # X = self.historical_df[['Race_ID', 'Course', 'Distance', 'Going', 'Surface', 'Class', 'number_of_turns']]
            # X = pd.get_dummies(X.drop('Race_ID', axis=1))
            # Z = linkage(X, 'ward')
            # clusters = fcluster(Z, 10, criterion='maxclust')
            # self.historical_df['Course_Cluster'] = clusters
            for col in cat_col:
                self.historical_df[col] = self.historical_df[col].astype('category')
            self.historical_df['Horse_ID'] = self.historical_df['Horse_ID'].astype(str)
            self.historical_df['Place'] = self.historical_df['Place'].astype(str)
            self.historical_df['Weight_Declared'] = self.historical_df['Weight_Declared'].astype(int)
            self.historical_df['Win_odds'] = self.historical_df['Win_odds'].astype(float)
            self.historical_df.to_parquet(DATAPATH / 'historical_model_input.parquet')

    @utils.logged
    def filling_null_db(self):
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
        self.historical_df = self.historical_df[self.historical_df['Place'] != '']
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
        self.historical_df = self.historical_df[self.historical_df['Weight_Declared'] != '---']
        self.historical_df['Weight'] = self.historical_df['Weight'].astype(int)
        self.historical_df['Weight_Declared'] = self.historical_df['Weight_Declared'].astype(int)
        self.historical_df.sort_values(['Date','Race_ID'], inplace=True)
        self.historical_df.to_parquet(DATAPATH / 'historical_model_input.parquet')

    def get_race_id(self):
        return self.historical_df['Race_ID'].max()+1

    def get_rs_id(self):
        return self.historical_df['RS_ID'].max()+1

    def get_sectional_id(self):
        return self.historical_df['Sectional_ID'].max()+1

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
        try:
            return re.search(r'(Class\s*\d+)', body).group(1).upper()
        except:
            return None

    def get_distance(self, body):
        return int(re.search(r'\s*(\d+)M\s*',body).group(1))

    def get_going(self, body):
        return re.search(r'Going\s*:\s*(.*?)\n',body).group(1)

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
        horse_detail_table['Horse_Code'] = horse_detail_table['Horse'].apply(lambda x: re.search(r'\((.*?)\)',x).group(1))
        horse_detail_table['Horse_Name'] = horse_detail_table['Horse'].apply(lambda x: x.split('(')[0])
        horse_detail_table.rename(
            {
                'Pla.': 'Place',
                'Horse No.': 'Horse_Number',
                'Act. Wt.': 'Weight',
                'Declar. Horse Wt.': 'Weight_Declared',
                'Dr.': 'Draw',
                'Win Odds': 'Win_odds'
            },
            axis=1,
            inplace=True
        )
        horse_detail_table['Draw'] = horse_detail_table['Draw'].astype(int)
        horse_detail_table.drop(['Horse','LBW', 'Running Position', 'Finish Time'], axis=1,inplace=True)
        return horse_detail_table

if __name__ == '__main__':
    cond = True
    while cond:
        try:
            processor = DBUPDATE_PROCESSOR()
            processor.head_to_result_pg()
            processor.update_db()
            processor.filling_null_db()
            cond = False
        except Exception as e:
            logging.info(traceback.format_exc())
            continue

