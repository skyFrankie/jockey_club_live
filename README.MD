# HK Horse Racing betting system

---
### This is a programmatic betting approach to archive positive returns on Horse Racing Gambling in HK

---

1. Please make sure the parameters inside ```credential.yaml``` are correctly filled. They should be the login account and password of HKJC account.
2. Inside the python file ```login_processor.py```, modify the function ```get_security_answer``` to fulfil the security questions of HKJC account.
``` python   
def get_security_answer():
    logging.info(f'Getting security answer...')
    securityquestion = self.driver.find_element(By.CSS_SELECTOR, 'div#ekbaSeqQuestion').text
    if securityquestion == '{security_question1}':
        security_answer = '{security_answer1}'
    elif securityquestion == '{security_question2}':
        security_answer = '{security_answer2}'
    else:
        security_answer = '{security_answer3}'
    return security_answer
```

3. Run the python file ```main.py```
