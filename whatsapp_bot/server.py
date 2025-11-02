import os
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")


# 1) Webhook Verification (Meta calls this ONCE)
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
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = msg["from"]

        # If user sends a button selection
        if msg["type"] == "interactive":
            selection_id = msg["interactive"]["button_reply"]["id"]
            handle_selection(sender, selection_id)
            return "OK", 200

        # If user sends text
        if msg["type"] == "text":
            user_text = msg["text"]["body"].strip().lower()

            if user_text:
                send_menu(sender)
            else:
                send_whatsapp_message(sender, "I didn't understand that 🤔\nType *menu* to see options again.")
        if msg["type"] == "location":
            lat = msg["location"]["latitude"]
            lon = msg["location"]["longitude"]

            forecast = get_weather_by_coordinates(lat, lon)
            send_whatsapp_message(sender, forecast)
            send_menu(sender)
            return "OK", 200
        
    except:
        pass

    return "OK", 200


# 3) Function to send WhatsApp messages back
def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    requests.post(url, headers=headers, json=payload)

def send_menu(to):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": "👋 *Welcome to GrowPak!* How can we help you today?"
            },
            "footer": {
                "text": "Select an option below:"
            },
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

def handle_selection(to, selection_id):
    if selection_id == "option_1":
        send_whatsapp_message(to, "🌾 *Crop Guidance Selected.*\n\n(We’ll help you choose best practices soon.)")

    elif selection_id == "option_2":
        send_whatsapp_message(to, "🦠 *Report Disease Selected.*\n\n(You’ll be able to send crop images & audio.)")

    elif selection_id == "option_3":
        send_whatsapp_message(to, "👨‍🌾 *Talk to Expert Selected.*\n\n(We'll connect you shortly.)")
    elif selection_id == "option_4":
        send_whatsapp_message(to, "🌦️ *Weather Forecast*\n\nPlease tap the button below to share your location:")
        request_location(to)
    # After response, show how to get menu again
    send_menu(to)

def request_location(to):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": "📍 Please share your location"},
            "action": {
                "buttons": [
                    {
                        "type": "location",
                        "reply": {"id": "send_location", "title": "Send Location"}
                    }
                ]
            }
        }
    }

    requests.post(url, headers=headers, json=payload)

def get_weather_by_coordinates(lat, lon):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    resp = requests.get(url).json()
    city = resp["city"]["name"]

    forecast_list = resp["list"][::8][:3]

    result = f"☁️ *Weather Forecast for {city}*\n\n"

    for entry in forecast_list:
        date = entry["dt_txt"].split(" ")[0]
        temp_min = entry["main"]["temp_min"]
        temp_max = entry["main"]["temp_max"]
        condition = entry["weather"][0]["description"].title()
        rain_chance = entry.get("pop", 0) * 100

        result += f"📅 {date}\n"
        result += f"🌡️ {temp_min:.0f}°C - {temp_max:.0f}°C\n"
        result += f"🌦️ {condition}\n"
        result += f"💧 Rain Chance: {rain_chance:.0f}%\n\n"

    return result


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
