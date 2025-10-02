# tools/weather.py
import os
import requests
from typing import Optional

def _get_api_key() -> Optional[str]:
    # 1) environment variable
    key = os.environ.get("OPENWEATHER_API_KEY") or os.environ.get("WEATHER_API_KEY")
    if key:
        return key
    # 2) streamlit secrets (if running inside Streamlit)
    try:
        import streamlit as st
        return st.secrets.get("OPENWEATHER_API_KEY") or st.secrets.get("WEATHER_API_KEY")
    except Exception:
        return None

def get_weather(location: str, api_key: Optional[str] = None) -> str:
    """
    Fetch current weather for `location` using OpenWeatherMap.
    - location: e.g. "Lagos" or "Lagos, NG"
    - api_key: optional override (otherwise looks in env / streamlit secrets)
    Returns: readable string or helpful error message.
    """
    if not location or not location.strip():
        return "Please provide a location (e.g. 'weather in Lagos')."

    api_key = api_key or _get_api_key()
    if not api_key:
        return (
            "Weather API key not found. "
            "Set OPENWEATHER_API_KEY in environment variables or add it to Streamlit secrets."
        )

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        weather_desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels = data["main"].get("feels_like")
        humidity = data["main"].get("humidity")
        wind = data.get("wind", {}).get("speed")

        return (
            f"{location.title()}: {weather_desc}. "
            f"Temp {temp:.1f}°C (feels like {feels:.1f}°C). "
            f"Humidity {humidity}% · Wind {wind} m/s."
        )
    except requests.HTTPError as e:
        return f"Unable to fetch weather for '{location}': {e}"
    except Exception as e:
        return f"Error fetching weather: {e}"
