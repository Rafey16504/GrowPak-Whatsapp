import os
import tempfile
import requests
from flask import Flask, request
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)

# ── WhatsApp / Meta credentials ────────────────────────────
VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN   = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID  = os.getenv("PHONE_NUMBER_ID")
OPENWEATHER_KEY  = os.getenv("OPENWEATHER_API_KEY")

# ── Import pipeline (loads all models at startup) ──────────
from pipeline import run_pipeline


# ═══════════════════════════════════════════════════════════
# 1. WEBHOOK VERIFICATION
# ═══════════════════════════════════════════════════════════
@app.get("/webhook")
def verify_webhook():
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


# ═══════════════════════════════════════════════════════════
# 2. RECEIVE MESSAGE
# ═══════════════════════════════════════════════════════════
@app.post("/webhook")
def receive_message():
    data = request.get_json()
    print("Incoming:", data)

    try:
        value = data["entry"][0]["changes"][0]["value"]

        if "messages" not in value:
            return "OK", 200

        msg    = value["messages"][0]
        sender = msg["from"]

        # ── List menu option selected ───────────────────────
        if msg["type"] == "interactive" and "list_reply" in msg["interactive"]:
            handle_selection(sender, msg["interactive"]["list_reply"]["id"])
            return "OK", 200

        # ── Location received (for weather) ─────────────────
        if msg["type"] == "location":
            lat = msg["location"]["latitude"]
            lon = msg["location"]["longitude"]
            send_whatsapp_message(sender, get_weather_by_coordinates(lat, lon))
            send_menu(sender)
            return "OK", 200

        # ── Voice/audio message → RAG pipeline ──────────────
        if msg["type"] == "audio":
            handle_voice_message(sender, msg["audio"])
            return "OK", 200

        # ── Text message → show menu ─────────────────────────
        if msg["type"] == "text":
            send_menu(sender)
            return "OK", 200

    except Exception as e:
        print("Error:", e)

    return "OK", 200


# ═══════════════════════════════════════════════════════════
# 3. VOICE MESSAGE HANDLER — runs the full pipeline
# ═══════════════════════════════════════════════════════════
def handle_voice_message(to: str, audio_obj: dict):
    """
    Download the WhatsApp voice note, run the full pipeline
    (STT → enhance → RAG → LLM → TTS), and send the audio reply.
    """
    media_id = audio_obj.get("id")
    if not media_id:
        send_whatsapp_message(to, "⚠️ Could not read your voice message. Please try again.")
        return

    # 1. Get media download URL from Meta
    media_url_resp = requests.get(
        f"https://graph.facebook.com/v20.0/{media_id}",
        headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
    )
    if media_url_resp.status_code != 200:
        send_whatsapp_message(to, "⚠️ Could not retrieve your voice message.")
        return

    download_url = media_url_resp.json().get("url")

    # 2. Download audio to a temp file
    audio_resp = requests.get(
        download_url,
        headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
    )
    suffix = ".ogg"  # WhatsApp voice notes are opus/ogg
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_resp.content)
        tmp_path = tmp.name

    # 3. Run pipeline
    try:
        send_whatsapp_message(to, "⏳ Processing your question...")
        result = run_pipeline(audio_path=tmp_path)
        final_answer  = result.get("final_answer", "")
        audio_out     = result.get("audio_response")

        # 4a. Send text answer
        if final_answer:
            send_whatsapp_message(to, final_answer)

        # 4b. Send audio reply
        if audio_out and os.path.exists(audio_out):
            send_whatsapp_audio(to, audio_out)
        else:
            if not final_answer:
                send_whatsapp_message(to, "⚠️ Could not generate a response. Please try again.")

    except Exception as e:
        print(f"[Pipeline error] {e}")
        send_whatsapp_message(to, "⚠️ Something went wrong while processing. Please try again.")
    finally:
        os.unlink(tmp_path)

    send_menu(to)


# ═══════════════════════════════════════════════════════════
# 4. SEND HELPERS
# ═══════════════════════════════════════════════════════════
def _wa_headers() -> dict:
    return {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

def _wa_url() -> str:
    return f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"


def send_whatsapp_message(to: str, message: str):
    requests.post(_wa_url(), headers=_wa_headers(), json={
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    })


def send_whatsapp_audio(to: str, audio_path: str):
    """Upload the MP3 to Meta and send it as an audio message."""
    # Step 1: Upload media
    upload_url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/media"
    with open(audio_path, "rb") as f:
        upload_resp = requests.post(
            upload_url,
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            files={"file": ("response.mp3", f, "audio/mpeg")},
            data={"messaging_product": "whatsapp", "type": "audio/mpeg"},
        )
    if upload_resp.status_code != 200:
        print(f"[Audio upload failed] {upload_resp.text}")
        return

    media_id = upload_resp.json().get("id")
    if not media_id:
        print("[Audio upload] No media ID returned.")
        return

    # Step 2: Send audio message
    requests.post(_wa_url(), headers=_wa_headers(), json={
        "messaging_product": "whatsapp",
        "to": to,
        "type": "audio",
        "audio": {"id": media_id},
    })


# ═══════════════════════════════════════════════════════════
# 5. MENU
# ═══════════════════════════════════════════════════════════
def send_menu(to: str):
    requests.post(_wa_url(), headers=_wa_headers(), json={
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": "👋 *Welcome to GrowPak!*\nChoose a service:"},
            "footer": {"text": "Grow smarter 🌱"},
            "action": {
                "button": "Select Option",
                "sections": [{
                    "title": "Available Services",
                    "rows": [
                        {"id": "option_1", "title": "🌾 Crop Guidance"},
                        {"id": "option_2", "title": "🦠 Report Disease"},
                        {"id": "option_3", "title": "👨‍🌾 Talk to Expert"},
                        {"id": "option_4", "title": "☁️ Weather Forecast"},
                    ],
                }],
            },
        },
    })


def handle_selection(to: str, selection_id: str):
    if selection_id == "option_1":
        send_whatsapp_message(
            to,
            "🌾 *Crop Guidance*\nSend me a voice note in Urdu or Punjabi with your question!"
        )
    elif selection_id == "option_2":
        send_whatsapp_message(
            to,
            "🦠 *Report Disease*\nDescribe the symptoms in a voice note and I'll help you identify it."
        )
    elif selection_id == "option_3":
        send_whatsapp_message(to, "👨‍🌾 *Talk to Expert*\nWe will connect you with an expert soon.")
    elif selection_id == "option_4":
        send_location_request(to)
        return

    send_menu(to)


def send_location_request(to: str):
    r = requests.post(_wa_url(), headers=_wa_headers(), json={
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "location_request_message",
            "body": {"text": "📍 Please share your location to get an accurate weather forecast."},
            "action": {"name": "send_location"},
        },
    })
    print("Location request response:", r.status_code, r.text)


# ═══════════════════════════════════════════════════════════
# 6. WEATHER
# ═══════════════════════════════════════════════════════════
def get_weather_by_coordinates(lat: float, lon: float) -> str:
    url  = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )
    data = requests.get(url).json()
    city = data["city"]["name"]

    daily = {}
    for entry in data["list"]:
        date = entry["dt_txt"].split(" ")[0]
        if date not in daily:
            daily[date] = {"temps": [], "rain": [], "conditions": []}
        daily[date]["temps"].append(entry["main"]["temp"])
        daily[date]["rain"].append(entry.get("pop", 0) * 100)
        daily[date]["conditions"].append(entry["weather"][0]["main"])

    icons = {"Rain": "🌧️", "Clouds": "⛅", "Clear": "☀️",
             "Drizzle": "🌦️", "Thunderstorm": "⛈️", "Snow": "❄️"}

    response = f"🌦️ *5-Day Weather for {city}*\n📍 Helps in deciding irrigation & crop care.\n\n"

    for i, (date, info) in enumerate(daily.items()):
        avg_min     = round(min(info["temps"]))
        avg_max     = round(max(info["temps"]))
        rain_chance = round(max(info["rain"]))
        condition   = max(set(info["conditions"]), key=info["conditions"].count)
        icon        = icons.get(condition, "🌍")
        day_label   = "Today" if i == 0 else datetime.strptime(date, "%Y-%m-%d").strftime("%a %d %b")

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

    response += "———————\n👨‍🌾 Tip: Weather changes often. Check daily for best crop decisions."
    return response


# ═══════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
