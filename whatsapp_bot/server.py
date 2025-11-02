import os
from flask import Flask, request
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

user_state = {}  # conversation context memory


# 1) Webhook Verification
@app.get("/webhook")
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


# 2) Receive messages
@app.post("/webhook")
def receive_message():
    data = request.get_json()
    print("Incoming:", data)

    try:
        value = data["entry"][0]["changes"][0]["value"]

        # Case: User sent location
        if "messages" in value:
            msg = value["messages"][0]
            sender = msg["from"]

            # If user clicked menu button
            if msg["type"] == "interactive":
                selection_id = msg["interactive"]["button_reply"]["id"]
                handle_selection(sender, selection_id)
                return "OK", 200

            # If user shared location
            if msg["type"] == "location":
                lat = msg["location"]["latitude"]
                lon = msg["location"]["longitude"]
                forecast = get_weather_by_coordinates(lat, lon)
                send_whatsapp_message(sender, forecast)
                send_menu(sender)
                return "OK", 200

            # If user sends text
            if msg["type"] == "text":
                send_menu(sender)
                return "OK", 200

    except Exception as e:
        print("Error:", e)

    return "OK", 200


# 3) Send text message
def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    requests.post(url, headers=headers, json=payload)


# 4) Send main menu with buttons
def send_menu(to):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": "👋 *Welcome to GrowPak!*\nChoose one of the options below:"},
            "footer": {"text": "Grow smarter 🌱"},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": "option_1", "title": "🌾 Crop Guidance"}},
                    {"type": "reply", "reply": {"id": "option_2", "title": "🦠 Report Disease"}},
                    {"type": "reply", "reply": {"id": "option_3", "title": "👨‍🌾 Talk to Expert"}},
                    {"type": "reply", "reply": {"id": "option_4", "title": "☁️ Weather Forecast"}}
                ]
            }
        }
    }

    requests.post(url, headers=headers, json=payload)


# 5) Handle menu selections
def handle_selection(to, selection_id):
    if selection_id == "option_1":
        send_whatsapp_message(to, "🌾 *Crop Guidance selected.*\nWe will guide you soon.")
    elif selection_id == "option_2":
        send_whatsapp_message(to, "🦠 *Report Disease selected.*\nYou will be able to send photos and audio soon.")
    elif selection_id == "option_3":
        send_whatsapp_message(to, "👨‍🌾 *Talk to Expert selected.*\nWe will connect you soon.")
    elif selection_id == "option_4":
        request_location(to)
        return
    
    send_menu(to)


# 6) Ask user to share location
def request_location(to):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": "📍 Please share your location"},
            "action": {
                "buttons": [
                    {"type": "location", "reply": {"id": "send_location", "title": "Send Location"}}
                ]
            }
        }
    }

    requests.post(url, headers=headers, json=payload)


# 7) Weather Forecast via Coordinates
def get_weather_by_coordinates(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    data = requests.get(url).json()

    city = data["city"]["name"]
    forecast_list = data["list"][::8][:3]

    response = f"🌦️ *3-Day Weather Forecast for {city}*\n\n"

    for entry in forecast_list:
        date = entry["dt_txt"].split(" ")[0]
        temp_min = entry["main"]["temp_min"]
        temp_max = entry["main"]["temp_max"]
        condition = entry["weather"][0]["description"].title()
        rain_chance = entry.get("pop", 0) * 100

        response += f"📅 *{date}*\n"
        response += f"🌡️ {temp_min:.0f}°C - {temp_max:.0f}°C\n"
        response += f"🌥️ {condition}\n"
        response += f"💧 Rain Chance: {rain_chance:.0f}%\n\n"

    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
