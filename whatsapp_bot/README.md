# GrowPak WhatsApp Bot — Setup Guide

This guide explains how to set up and run the GrowPak WhatsApp bot locally using the WhatsApp Cloud API, Flask, and Ngrok. The bot displays a menu of options when the user sends a message.

## Requirements

| Requirement             | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| Meta Developer Account  | [Meta Developers](https://developers.facebook.com)                         |
| Python 3.10+            | Needed to run the backend server                                           |
| Ngrok (Free)            | Used to expose local server publicly                                       |
| Personal WhatsApp Account | Used to test the bot                                                     |
| No business phone number required | We use Meta's built-in test phone number                                |

---

## 1. Create WhatsApp Cloud API App on Meta

1. Go to [Meta Developers](https://developers.facebook.com).
2. Click **Create App**.
3. Choose **Business App**.
4. Finish setup steps.
5. On the left sidebar, go to **WhatsApp**.
6. Click **Get Started**.

A sandbox test environment will appear containing:
- **Test WhatsApp Phone Number**
- **Phone Number ID**
- **Temporary Access Token** (expires every 24 hours)

## 2. Create `.env` File

In your project folder, create a file named `.env`:

```env
VERIFY_TOKEN=your_verify_token_here
WHATSAPP_TOKEN=your_meta_access_token
PHONE_NUMBER_ID=your_phone_number_id
```

## 3. Install Required Dependencies

```
pip install flask python-dotenv requests
```

## 4. Python Server Behavior Summary

The server performs the following actions:

| Action               | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| Webhook Verification | Confirms server ownership during setup                                 |
| Receive Messages     | Listens for messages from users                                        |
| Send Messages        | Replies back via WhatsApp Cloud API                                    |
| Show Menu            | Sends a welcome message and button menu when the user types `hello`, `hi`, `start`, or `menu` |

The menu includes interactive reply buttons:

- 🌾 Crop Guidance
- 🦠 Report Disease
- 👨‍🌾 Talk to Expert

The bot responds to the selected choice and instructs the user to type `menu` to view options again.

---

## 5. Run the Flask Server

Run the Flask server using the following command:

```bash
python server.py    (Running on http://127.0.0.1:3000)
```

## 6. Run the ngrok server

Run the ngrok server using the following command:

```bash
ngrok http 3000    (https://21ef59c7a1e1.ngrok-free.app)
```

## 7. Configure Webhook in meta dashboard

Go to: WhatsApp → Configuration -> Webhook → click Configure

Set the following:

- Callback URL	: https://YOUR-NGROK-URL/webhook
- Verify Token:	same value used in .env

- Click Verify & Save

-Click Manage Webhook Fields -->Turn on: messages

## 8. System Flow Summary

| Step                            | Description                           |
| ------------------------------- | ------------------------------------- |
| User sends message              | WhatsApp forwards it to your webhook  |
| Flask parses message            | Determines if text or button selected |
| Server generates reply          | Based on greeting or menu option      |
| Response sent through Cloud API | Delivered instantly to WhatsApp       |
