# -*- coding: utf-8 -*-
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import get_key
from httpx import Response, get, post

logger = logging.getLogger(__name__)

def find_dotenv() -> str:
    for candidate in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]:
        if candidate.exists():
            return str(candidate)
    return ".env"

DOTENV_PATH: str = find_dotenv()
TOKEN: Optional[str] = os.getenv("TOKEN")
if not TOKEN and os.path.exists(DOTENV_PATH):
    try:
        TOKEN = get_key(DOTENV_PATH, "TOKEN")
    except Exception:
        TOKEN = None

# Locate subscribers file (check root and cwd)
SUBSCRIBERS_CANDIDATE = Path(__file__).resolve().parent.parent / "subscribers.pickle"
if SUBSCRIBERS_CANDIDATE.exists():
    SUBSCRIBERS_FILE: str = str(SUBSCRIBERS_CANDIDATE)
else:
    SUBSCRIBERS_FILE: str = "subscribers.pickle"

BASE_URL: str = f"https://api.telegram.org/bot{TOKEN}" if TOKEN else ""


def get_updates() -> Optional[Dict]:
    if not TOKEN:
        raise ValueError("Telegram Bot TOKEN is not configured in .env")
    url: str = f"{BASE_URL}/getUpdates"
    response: Response = post(url)
    response_body: Dict = response.json()
    if not response_body.get("ok"):
        raise Exception(f"Failed to fetch updates: {response_body}")
    return response_body


def get_filtered_messages(response_body: Dict, type: str) -> Iterable[Dict]:
    return filter(lambda update: type in update.get("message", {}), response_body.get("result", []))


def update_subscribers() -> List:
    if not TOKEN:
        print("⚠️ Telegram TOKEN is missing. Set TOKEN in .env to update subscribers.")
        return []
    try:
        updates: Dict = get_updates()
        messages: List[Dict] = list(get_filtered_messages(updates, "text"))
        fetched_subscribers = [
            update["message"]["chat"]["id"]
            for update in messages
            if update.get("message", {}).get("text") == "/start"
        ]
        if not os.path.exists(SUBSCRIBERS_FILE):
            with open(SUBSCRIBERS_FILE, "wb") as file:
                pickle.dump(list(), file)
        with open(SUBSCRIBERS_FILE, "rb") as file:
            old_subscribers: List[int] = pickle.load(file)
            new_subscribers: List[int] = list(
                filter(lambda chat_id: chat_id not in old_subscribers, fetched_subscribers)
            )
            old_subscribers.extend(new_subscribers)
            for chat_id in new_subscribers:
                message: str = "Welcome to the Crime Patrol / Criminal Detection Alert Bot!"
                url: str = f"{BASE_URL}/sendMessage"
                query_parameters = {"chat_id": chat_id, "text": message}
                response: Response = get(url, params=query_parameters)
                response_body: Dict = response.json()
                if not response_body.get("ok"):
                    print(f"Failed to send welcome message to chat {chat_id}")
        with open(SUBSCRIBERS_FILE, "wb") as file:
            pickle.dump(list(old_subscribers), file)
        print(f"Subscribers updated. Total subscribers: {len(old_subscribers)}")
        return old_subscribers
    except Exception as e:
        print(f"⚠️ Error updating subscribers: {e}")
        return []


def load_subscribers() -> List[int]:
    if not os.path.exists(SUBSCRIBERS_FILE):
        return []
    try:
        with open(SUBSCRIBERS_FILE, "rb") as file:
            subscribers: List[int] = pickle.load(file)
        return subscribers
    except Exception as e:
        logger.warning(f"Failed to load subscribers from {SUBSCRIBERS_FILE}: {e}")
        return []


def send_message(message: str) -> None:
    """Function to send a plaintext alert message"""
    if not TOKEN:
        logger.info(f"[Telegram Alert - No TOKEN configured]: {message}")
        return

    subscribers: List[int] = load_subscribers()
    if not subscribers:
        logger.info(f"[Telegram Alert - No Subscribers]: {message}")
        return

    url: str = f"{BASE_URL}/sendMessage"
    for chat_id in subscribers:
        try:
            query_parameters: Dict = {"chat_id": chat_id, "text": message}
            response: Response = get(url, params=query_parameters, timeout=5.0)
            response_body: Dict = response.json()
            if not response_body.get("ok"):
                logger.warning(f"Failed to send message to chat {chat_id}: {response_body}")
            else:
                print(f"📲 Telegram alert sent to chat {chat_id}")
        except Exception as e:
            logger.warning(f"Error sending Telegram message to {chat_id}: {e}")


def send_video(video: str, spoiler: bool = True) -> None:
    """Function to send a video alert message"""
    if not TOKEN:
        logger.info(f"[Telegram Alert - No TOKEN configured]: (Video: {video})")
        return

    subscribers: List[int] = load_subscribers()
    if not subscribers:
        logger.info(f"[Telegram Alert - No Subscribers]: (Video: {video})")
        return

    if not os.path.exists(video):
        logger.warning(f"Video file not found: {video}")
        return

    url: str = f"{BASE_URL}/sendVideo"
    try:
        with open(video, "rb") as file:
            for chat_id in subscribers:
                try:
                    file.seek(0)
                    files: Dict = {"video": file}
                    request_body: Dict = {
                        "chat_id": chat_id,
                        "has_spoiler": spoiler,
                        "caption": "🚨 Criminal Face Detection Alert Recording",
                    }
                    response: Response = post(url, data=request_body, files=files, timeout=30.0)
                    response_body: Dict = response.json()
                    if not response_body.get("ok"):
                        logger.warning(f"Failed to send video to {chat_id}: {response_body}")
                    else:
                        print(f"📲 Telegram video alert sent to chat {chat_id}")
                except Exception as e:
                    logger.warning(f"Error sending video to {chat_id}: {e}")
    except Exception as e:
        logger.warning(f"Error reading video {video}: {e}")


def main() -> None:
    update_subscribers()
    print("1 - Send message")
    print("2 - Send video")
    option: int = int(input("Enter an option: "))
    if option == 1:
        message: str = input("Enter the message to send: ")
        send_message(message)
    elif option == 2:
        video: str = input("Enter the video path: ")
        send_video(video)


if __name__ == "__main__":
    main()
