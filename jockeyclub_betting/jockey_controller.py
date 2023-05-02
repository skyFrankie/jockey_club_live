from jockeyclub_betting.login_processor import LoginProcessor
from jockeyclub_betting import utils

class JockeyController:
    def __init__(self, credential_path):
        creds = utils.read_credentials(credential_path)
        self.login_bot = LoginProcessor(
            acc=creds.get('account', None),
            pwd=creds.get('pwd', None),
            jc_uri=creds.get('jc_uri', None)
        )
        self.driver = self.login_bot.driver

controller = JockeyController(utils.PROJECT_PATH/'credentials.yaml')