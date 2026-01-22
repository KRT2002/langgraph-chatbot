"""
Weather information tool using OpenWeatherMap API.
"""

import requests
from langchain_core.tools import tool

from langgraph_chatbot.config import settings
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


@tool
def get_weather(city: str, units: str = "metric") -> dict:
    """
    Get current weather information for a city.
    
    Parameters
    ----------
    city : str
        Name of the city
    units : str, optional
        Temperature units: 'metric' (Celsius), 'imperial' (Fahrenheit), by default 'metric'
        
    Returns
    -------
    dict
        Weather information including temperature, description, humidity, etc.
        
    Examples
    --------
    >>> get_weather("London")
    {'status': 'success', 'city': 'London', 'temperature': 15.5, 'description': 'cloudy', ...}
    """
    try:
        logger.info(f"Fetching weather for {city} in {units} units")
        
        if not settings.openweather_api_key:
            logger.error("OpenWeather API key not configured")
            return {
                "status": "error",
                "error_type": "api_key_missing",
                "message": "Weather API key not configured. Please add OPENWEATHER_API_KEY to .env file"
            }
        
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": settings.openweather_api_key,
            "units": units
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        result = {
            "status": "success",
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "units": "°C" if units == "metric" else "°F"
        }
        
        logger.info(f"Weather data retrieved for {city}: {result['temperature']}{result['units']}")
        return result
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"City not found: {city}")
            return {
                "status": "error",
                "error_type": "city_not_found",
                "message": f"City '{city}' not found. Please check the spelling or please take action on basis of this error: {str(e)}"
            }
        logger.error(f"HTTP error fetching weather: {str(e)}")
        return {
            "status": "error",
            "error_type": "api_error",
            "message": f"Failed to fetch weather: {str(e)}"
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching weather: {str(e)}")
        return {
            "status": "error",
            "error_type": "network_error",
            "message": f"Network error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Unexpected error in get_weather: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": str(e)
        }