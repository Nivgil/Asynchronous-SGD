from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib


def send_notification(msg_text_1, msg_text_2):
    fromaddr = "technion.212@gmail.com"
    toaddr = "giladiniv@cs.technion.ac.il"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Code Notification"

    body = msg_text_1 + '\n\n\n'
    msg.attach(MIMEText(body, 'plain'))
    body = dict_to_table(msg_text_2)
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login("technion.212", "nivgiladi218")
    text = msg.as_string()
    print('Sending Email Notification')
    server.sendmail(fromaddr, toaddr, text)


def dict_to_table(dic):
    str = '{:<8} {:<25} {:<10}\n'.format('Number', 'Name', 'Value')
    for num, v in enumerate(dic.items()):
        label, k = v
        str = str + '{:<8} {:<25} {:<10}\n'.format(num, label, k)
    return str