import requests
import streamlit as st

# Get API key securely from Streamlit secrets
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]

def get_weather(city: str, country_code: str = None) -> str:
    """
    Fetch current weather data for a given city (and optional country code).
    Returns a simple string summary.
    
    Example:
        get_weather("Lagos", "NG")
    """
    try:
        # Build query (with or without country code)
        if country_code:
            query = f"{city},{country_code}"
        else:
            query = city

        url = f"http://api.openweathermap.org/data/2.5/weather?q={query}&appid={OPENWEATHER_API_KEY}&units=metric"

        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return f"⚠️ Could not fetch weather for {city}. Error: {data.get('message', 'Unknown error')}"

        # Extract useful details
        weather_main = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]

        return (
            f"🌤 Weather in {city}:\n"
            f"- Condition: {weather_main}\n"
            f"- Temperature: {temp}°C (feels like {feels_like}°C)\n"
            f"- Humidity: {humidity}%"
        )

    except Exception as e:
        return f"⚠️ An error occurred: {str(e)}"
