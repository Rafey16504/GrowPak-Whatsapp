import os
from flask import Flask, request
import requests
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

app = Flask(__name__)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# 1) VERIFY WEBHOOK
@app.get("/webhook")
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


# 2) RECEIVE MESSAGE
@app.post("/webhook")
def receive_message():
    data = request.get_json()
    print("Incoming:", data)

    try:
        value = data["entry"][0]["changes"][0]["value"]

        if "messages" in value:
            msg = value["messages"][0]
            sender = msg["from"]

            # --- LIST MENU OPTION SELECTED ---
            if msg["type"] == "interactive" and "list_reply" in msg["interactive"]:
                selection_id = msg["interactive"]["list_reply"]["id"]
                handle_selection(sender, selection_id)
                return "OK", 200

            # --- LOCATION RECEIVED ---
            if msg["type"] == "location":
                lat = msg["location"]["latitude"]
                lon = msg["location"]["longitude"]
                forecast = get_weather_by_coordinates(lat, lon)
                send_whatsapp_message(sender, forecast)
                send_menu(sender)
                return "OK", 200

            # --- TEXT MESSAGE ---
            if msg["type"] == "text":
                send_menu(sender)
                return "OK", 200

    except Exception as e:
        print("Error:", e)

    return "OK", 200


# 3) SEND TEXT MESSAGE
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


# 4) SEND LIST MENU
def send_menu(to):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": "👋 *Welcome to GrowPak!*\nChoose a service:"},
            "footer": {"text": "Grow smarter 🌱"},
            "action": {
                "button": "Select Option",
                "sections": [
                    {
                        "title": "Available Services",
                        "rows": [
                            {"id": "option_1", "title": "🌾 Crop Guidance"},
                            {"id": "option_2", "title": "🦠 Report Disease"},
                            {"id": "option_3", "title": "👨‍🌾 Talk to Expert"},
                            {"id": "option_4", "title": "☁️ Weather Forecast"}
                        ]
                    }
                ]
            }
        }
    }

    requests.post(url, headers=headers, json=payload)


# 5) MENU SELECTION HANDLER
def handle_selection(to, selection_id):
    if selection_id == "option_1":
        send_whatsapp_message(to, "🌾 *Crop Guidance*\nGuidance features coming soon.")
    elif selection_id == "option_2":
        send_whatsapp_message(to, "🦠 *Report Disease*\nYou may send plant images or voice soon.")
    elif selection_id == "option_3":
        send_whatsapp_message(to, "👨‍🌾 *Talk to Expert*\nWe will connect you soon.")
    elif selection_id == "option_4":
        send_location_request(to)
        return

    send_menu(to)

def send_location_request(to):
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
            "type": "location_request_message",
            "body": {
                "text": "📍 Please share your location to get an accurate weather forecast."
            },
            "action": {
                "name": "send_location"   # ✅ This was missing
            }
        }
    }

    r = requests.post(url, headers=headers, json=payload)
    print("SEND_LOCATION_REQUEST RESPONSE:", r.status_code, r.text)



# 6) ASK USER LOCATION FOR WEATHER
def request_location(to):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "location_request",
            "body": {
                "text": "📍 Please share your location so I can provide accurate weather information."
            }
        }
    }

    r = requests.post(url, headers=headers, json=payload)
    print("LOCATION REQUEST RESPONSE:", r.status_code, r.text)



# 7) WEATHER API
def get_weather_by_coordinates(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    data = requests.get(url).json()

    city = data["city"]["name"]
    forecast_list = data["list"]

    daily = {}
    for entry in forecast_list:
        date = entry["dt_txt"].split(" ")[0]
        temp = entry["main"]["temp"]
        pop = entry.get("pop", 0) * 100
        condition = entry["weather"][0]["main"]

        if date not in daily:
            daily[date] = {"temps": [], "rain": [], "conditions": []}

        daily[date]["temps"].append(temp)
        daily[date]["rain"].append(pop)
        daily[date]["conditions"].append(condition)

    response = f"🌦️ *5-Day Weather for {city}*\n"
    response += f"📍 Weather helps in deciding irrigation & crop care.\n\n"

    weather_symbols = {
        "Rain": "🌧️",
        "Clouds": "⛅",
        "Clear": "☀️",
        "Drizzle": "🌦️",
        "Thunderstorm": "⛈️",
        "Snow": "❄️"
    }

    for i, (date, info) in enumerate(daily.items()):
        avg_min = round(min(info["temps"]))
        avg_max = round(max(info["temps"]))
        rain_chance = round(max(info["rain"]))
        most_condition = max(set(info["conditions"]), key=info["conditions"].count)
        icon = weather_symbols.get(most_condition, "🌍")

        day_label = "Today" if i == 0 else datetime.strptime(date, "%Y-%m-%d").strftime("%a %d %b")

        if rain_chance > 60:
            advice = "🌧️ *Do NOT irrigate today.* Rain expected."
        elif rain_chance > 30:
            advice = "🌦️ *Irrigate only if needed.* Chance of rain."
        else:
            advice = "💧 *Safe to irrigate.* No rain expected."

        response += (
            f"📅 *{day_label}*  {icon}\n"
            f"🌡️ Temp: *{avg_min}°C - {avg_max}°C*\n"
            f"💧 Rain Chance: *{rain_chance}%*\n"
            f"🔹 *Advice:* {advice}\n\n"
        )

        if i == 4:
            break

    response += "———————\n"
    response += "👨‍🌾 Tip: Weather changes often. Check daily for best crop decisions."

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
