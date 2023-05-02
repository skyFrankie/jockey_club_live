from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
import time
import logging
import jockeyclub_betting.utils as utils


@utils.logged
class LoginProcessor:
    def __init__(self, acc, pwd, jc_uri):
        self.acc = acc
        self.pwd = pwd
        self.jc_uri = jc_uri
        try:
            validation = [self.acc, self.pwd, self.jc_uri]
            if not all(validation) or validation == []:
                raise ValueError('Missing essential login parameters.')
            self.driver = self.start_driver()
            self.login_process()
        except Exception as e:
            logging.exception(f'{e}', exc_info=True)

    @utils.logged
    def start_driver(self):
        logging.info('Starting chrome driver...')
        return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    @utils.logged
    def init_jc(self):
        self.driver.maximize_window()
        self.driver.get(self.jc_uri)

    def waiting_pop(self, targets):
        for target in targets:
            WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, target)))

    def fill_in_box(self, element, sending, keys=False):
        logging.info(f'Filling in {element}...')
        box = self.driver.find_element(By.CSS_SELECTOR, element)
        if not keys:
            ActionChains(self.driver).move_to_element(box).click().send_keys(sending).perform()
        else:
            ActionChains(self.driver).move_to_element(box).click().send_keys(sending).send_keys(Keys.RETURN).perform()

    @utils.logged
    def get_security_answer(self):
        logging.info(f'Getting security answer...')
        securityquestion = self.driver.find_element(By.CSS_SELECTOR, 'div#ekbaSeqQuestion').text
        if securityquestion == '你兒時住在哪區?':
            security_answer = 'Taipo'
        elif securityquestion == '你最好的朋友叫甚麼名字?':
            security_answer = 'Edward'
        else:
            security_answer = 'Math'
        return security_answer

    def click_button(self,target):
        webloginbtn = self.driver.find_element(By.CSS_SELECTOR, target)
        webloginbtn.click()

    @utils.logged
    def login_process(self):
        self.init_jc()
        logging.info('Running login process')
        time.sleep(0.5)
        self.waiting_pop(
            ['input#passwordInput1.accInfoInputField','input#account.accInfoInputField','div#loginButton']
        )
        time.sleep(0.5)
        self.fill_in_box('input#account.accInfoInputField',self.acc)
        time.sleep(0.5)
        self.fill_in_box('input#passwordInput1',self.pwd)
        time.sleep(0.5)
        self.click_button('div#loginButton')
        time.sleep(0.5)
        self.waiting_pop(['div#ekbaSeqQuestion'])
        time.sleep(0.5)
        self.fill_in_box('input#ekbaDivInput.text', self.get_security_answer(), True)
        time.sleep(0.5)
        self.waiting_pop(['div#disclaimerProceed'])
        time.sleep(0.5)
        self.click_button('div#disclaimerProceed')
        time.sleep(0.5)
        logging.info('Finished login process')
