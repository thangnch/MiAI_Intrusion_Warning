import telegram

def send_telegram(photo_path="alert.png"):
    try:
        my_token = "5404626616:AAGqQBl1bdUf0_cj2SQrFdZPiHoSy6VUVZg"
        bot = telegram.Bot(token=my_token)
        bot.sendPhoto(chat_id="1246123900", photo=open(photo_path, "rb"), caption="Có xâm nhập, nguy hiêm!")
    except Exception as ex:
        print("Can not send message telegram ", ex)

    print("Send sucess")