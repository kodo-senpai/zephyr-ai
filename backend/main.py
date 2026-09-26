import asyncio

import math

import time

import os

from contextlib import asynccontextmanager

from typing import Any

from pydantic import BaseModel, Field



import httpx

from fastapi import FastAPI, HTTPException, Query

from fastapi.middleware.cors import CORSMiddleware



# ============================================================

# CONFIGURATION

# ============================================================



OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"



CACHE_SECONDS = 60

INTELLIGENCE_CACHE_SECONDS = 300







# ============================================================

# LOCAL ENV LOADER

# ============================================================



def load_local_env() -> None:

    """Load simple KEY=VALUE entries from backend/.env if present."""

    env_path = os.path.join(os.path.dirname(__file__), ".env")

    if not os.path.exists(env_path):

        return

    try:

        with open(env_path, "r", encoding="utf-8") as handle:

            for raw in handle:

                line = raw.strip()

                if not line or line.startswith("#") or "=" not in line:

                    continue

                key, value = line.split("=", 1)

                key = key.strip()

                value = value.strip().strip('"').strip("'")

                if key:

                    os.environ.setdefault(key, value)

    except OSError:

        pass



load_local_env()



NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_APIKEY")

NVIDIA_API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")

NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "")
NVIDIA_MODELS_URL = os.getenv("NVIDIA_MODELS_URL", "https://integrate.api.nvidia.com/v1/models")



# ============================================================

# LOCATIONS

# ============================================================



LOCATIONS = {

    "chamoli": {

        "name": "Chamoli / Joshimath",

        "city": "Joshimath",

        "state": "Uttarakhand",

        "latitude": 30.5669,

        "longitude": 79.5640,

    },

    "shimla": {

        "name": "Shimla",

        "city": "Shimla",

        "state": "Himachal Pradesh",

        "latitude": 31.1048,

        "longitude": 77.1734,

    },

    "guwahati": {

        "name": "Guwahati",

        "city": "Guwahati",

        "state": "Assam",

        "latitude": 26.1445,

        "longitude": 91.7362,

    },

    "mumbai": {

        "name": "Mumbai",

        "city": "Mumbai",

        "state": "Maharashtra",

        "latitude": 19.0760,

        "longitude": 72.8777,

    },

    "puri": {

        "name": "Puri",

        "city": "Puri",

        "state": "Odisha",

        "latitude": 19.8135,

        "longitude": 85.8312,

    },

}



# ============================================================

# WMO WEATHER CODES

# ============================================================



WEATHER_CODES = {

    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",

    45: "Fog", 48: "Depositing rime fog",

    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",

    56: "Light freezing drizzle", 57: "Dense freezing drizzle",

    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",

    66: "Light freezing rain", 67: "Heavy freezing rain",

    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",

    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",

    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",

    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",

}



# ============================================================

# GLOBAL STATE

# ============================================================



http_client: httpx.AsyncClient | None = None



cache = {"timestamp": 0.0, "data": None}

intelligence_cache: dict[str, dict[str, Any]] = {}

cache_lock = asyncio.Lock()



# ============================================================

# FASTAPI LIFESPAN / APP

# ============================================================



@asynccontextmanager

async def lifespan(app: FastAPI):

    global http_client

    http_client = httpx.AsyncClient(

        timeout=httpx.Timeout(20.0, connect=10.0),

        headers={"User-Agent": "ZephyrAI/2.0 weather-dashboard"},

        follow_redirects=True,

    )



    print("==========================================")

    print("        ZEPHYR AI BACKEND STARTED")

    print("==========================================")

    print("Weather source : Open-Meteo")

    print("Intelligence   : ECMWF + Ensemble")

    print("API server     : http://127.0.0.1:8000")

    print("==========================================")



    yield



    await http_client.aclose()

    http_client = None





app = FastAPI(

    title="Zephyr AI Weather Intelligence API",

    description="Zephyr AI weather intelligence with forecast visualization, risk analysis and grounded AI analyst.",

    version="3.0.0",

    lifespan=lifespan,

)



app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://127.0.0.1:3000",

        "http://localhost:3000",

        "http://127.0.0.1:5500",

        "http://localhost:5500",

    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



# ============================================================

# BASIC HELPERS

# ============================================================



def weather_description(code: int | None) -> str:

    if code is None:

        return "Unknown"

    return WEATHER_CODES.get(code, f"Weather condition code {code}")





def safe_float(value: Any) -> float | None:

    try:

        if value is None:

            return None

        result = float(value)

        if not math.isfinite(result):

            return None

        return result

    except (TypeError, ValueError):

        return None





def validate_coordinates(latitude: float, longitude: float) -> None:

    if not -90 <= latitude <= 90:

        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:

        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180.")





def get_risk_level(

    weather_code: int | None,

    precipitation_probability: float | None,

    precipitation: float | None,

    wind_gust: float | None = None,

) -> str:

    if weather_code in [95, 96, 99]:

        return "HIGH"

    if weather_code in [65, 67, 82, 86]:

        return "HIGH"

    if wind_gust is not None and wind_gust >= 60:

        return "HIGH"

    if precipitation_probability is not None and precipitation_probability >= 70:

        return "MODERATE"

    if precipitation is not None and precipitation >= 10:

        return "MODERATE"

    if weather_code in [63, 81, 80]:

        return "MODERATE"

    return "LOW"





def extract_hourly_forecast(hourly: dict[str, Any], limit: int = 24) -> list[dict[str, Any]]:

    times = hourly.get("time", [])

    temperatures = hourly.get("temperature_2m", [])

    rain_probability = hourly.get("precipitation_probability", [])

    precipitation = hourly.get("precipitation", [])

    weather_codes = hourly.get("weather_code", [])

    wind_gusts = hourly.get("wind_gusts_10m", [])

    cloud_cover = hourly.get("cloud_cover", [])

    wind_speed = hourly.get("wind_speed_10m", [])

    wind_direction = hourly.get("wind_direction_10m", [])



    result = []

    limit = min(limit, len(times))



    for i in range(limit):

        code = weather_codes[i] if i < len(weather_codes) else None

        result.append({

            "time": times[i],

            "temperature_c": temperatures[i] if i < len(temperatures) else None,

            "precipitation_probability": rain_probability[i] if i < len(rain_probability) else None,

            "precipitation_mm": precipitation[i] if i < len(precipitation) else None,

            "wind_gust_kmh": wind_gusts[i] if i < len(wind_gusts) else None,

            "cloud_cover_percent": cloud_cover[i] if i < len(cloud_cover) else None,

            "wind_speed_kmh": wind_speed[i] if i < len(wind_speed) else None,

            "wind_direction_deg": wind_direction[i] if i < len(wind_direction) else None,

            "weather_code": code,

            "condition": weather_description(code),

        })



    return result



# ============================================================

# FORECAST FETCH

# ============================================================



async def fetch_weather_coordinates(

    latitude: float,

    longitude: float,

    name: str | None = None,

    city: str | None = None,

    state: str | None = None,

) -> dict[str, Any]:

    if http_client is None:

        raise RuntimeError("HTTP client is not initialized.")



    validate_coordinates(latitude, longitude)



    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current": ",".join([

            "temperature_2m", "relative_humidity_2m", "apparent_temperature",

            "precipitation", "rain", "cloud_cover", "pressure_msl",

            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "weather_code",

        ]),

        "hourly": ",".join([

            "temperature_2m", "precipitation_probability", "precipitation",

            "rain", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "weather_code",

        ]),

        "daily": ",".join([

            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",

            "precipitation_probability_max", "weather_code", "wind_gusts_10m_max",

        ]),

        "forecast_days": 7,

        "timezone": "auto",

    }



    response = await http_client.get(OPEN_METEO_URL, params=params)

    response.raise_for_status()

    data = response.json()



    current = data.get("current", {})

    current_units = data.get("current_units", {})

    hourly = data.get("hourly", {})

    daily = data.get("daily", {})



    weather_code = current.get("weather_code")

    precipitation_probability = (

        hourly.get("precipitation_probability", [None])[0]

        if hourly.get("precipitation_probability") else None

    )



    risk = get_risk_level(

        weather_code,

        precipitation_probability,

        current.get("precipitation"),

        current.get("wind_gusts_10m"),

    )



    location_name = name or city or "Custom location"



    return {

        "id": "custom",

        "location": {

            "name": location_name,

            "city": city,

            "state": state,

            "latitude": data.get("latitude", latitude),

            "longitude": data.get("longitude", longitude),

            "elevation_m": data.get("elevation"),

        },

        "source": {

            "provider": "Open-Meteo",

            "type": "Forecast API",

            "synthetic_data": False,

        },

        "observation": {

            "time": current.get("time"),

            "timezone": data.get("timezone"),

        },

        "current": {

            "temperature_c": current.get("temperature_2m"),

            "temperature_unit": current_units.get("temperature_2m", "°C"),

            "feels_like_c": current.get("apparent_temperature"),

            "humidity_percent": current.get("relative_humidity_2m"),

            "wind_speed_kmh": current.get("wind_speed_10m"),

            "wind_direction_deg": current.get("wind_direction_10m"),

            "wind_gust_kmh": current.get("wind_gusts_10m"),

            "precipitation_mm": current.get("precipitation"),

            "rain_mm": current.get("rain"),

            "cloud_cover_percent": current.get("cloud_cover"),

            "pressure_hpa": current.get("pressure_msl"),

            "weather_code": weather_code,

            "condition": weather_description(weather_code),

            "risk_level": risk,

        },

        "hourly": extract_hourly_forecast(hourly, 24),

        "daily": {

            "time": daily.get("time", []),

            "temperature_max_c": daily.get("temperature_2m_max", []),

            "temperature_min_c": daily.get("temperature_2m_min", []),

            "precipitation_sum_mm": daily.get("precipitation_sum", []),

            "precipitation_probability_max": daily.get("precipitation_probability_max", []),

            "wind_gust_max_kmh": daily.get("wind_gusts_10m_max", []),

            "weather_code": daily.get("weather_code", []),

        },

    }





async def fetch_location(location_id: str, location: dict[str, Any]) -> dict[str, Any]:

    result = await fetch_weather_coordinates(

        location["latitude"], location["longitude"],

        name=location["name"], city=location["city"], state=location["state"],

    )

    result["id"] = location_id

    return result



# ============================================================

# GEOCODING

# ============================================================



async def search_locations(query: str) -> list[dict[str, Any]]:

    if http_client is None:

        raise RuntimeError("HTTP client is not initialized.")



    query = query.strip()

    if len(query) < 2:

        return []



    response = await http_client.get(

        GEOCODING_URL,

        params={

            "name": query,

            "count": 8,

            "language": "en",

            "format": "json",

        },

    )

    response.raise_for_status()

    data = response.json()



    results = []

    for item in data.get("results", []):

        results.append({

            "name": item.get("name"),

            "city": item.get("name"),

            "state": item.get("admin1"),

            "country": item.get("country"),

            "country_code": item.get("country_code"),

            "latitude": item.get("latitude"),

            "longitude": item.get("longitude"),

            "timezone": item.get("timezone"),

            "elevation_m": item.get("elevation"),

        })



    return results



# ============================================================

# ECMWF + ENSEMBLE INTELLIGENCE

# ============================================================



async def fetch_ecmwf(latitude: float, longitude: float) -> dict[str, Any]:

    if http_client is None:

        raise RuntimeError("HTTP client is not initialized.")



    response = await http_client.get(

        ECMWF_URL,

        params={

            "latitude": latitude,

            "longitude": longitude,

            "hourly": ",".join([

                "temperature_2m", "precipitation", "rain", "weather_code",

                "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "cape",

            ]),

            "forecast_hours": 48,

            "timezone": "auto",

        },

    )

    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        if not data:
            raise RuntimeError("ECMWF API returned an empty response list.")
        data = data[0]
    if not isinstance(data, dict):
        raise RuntimeError("ECMWF API returned an unexpected response format.")
    return data





async def fetch_ensemble(latitude: float, longitude: float) -> dict[str, Any]:

    if http_client is None:

        raise RuntimeError("HTTP client is not initialized.")



    response = await http_client.get(

        ENSEMBLE_URL,

        params={

            "latitude": latitude,

            "longitude": longitude,

            "models": "ecmwf_ifs025_ensemble",

            "hourly": ",".join([

                "temperature_2m", "precipitation", "wind_gusts_10m",

            ]),

            "forecast_hours": 24,

            "timezone": "auto",

        },

    )

    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        if not data:
            raise RuntimeError("Ensemble API returned an empty response list.")
        data = data[0]
    if not isinstance(data, dict):
        raise RuntimeError("Ensemble API returned an unexpected response format.")
    return data





def ensemble_member_values(hourly: dict[str, Any], variable: str, index: int) -> list[float]:

    values = []

    prefix = f"{variable}_member"

    for key, series in hourly.items():

        if key.startswith(prefix) and isinstance(series, list) and index < len(series):

            value = safe_float(series[index])

            if value is not None:

                values.append(value)

    return values





def mean(values: list[float]) -> float | None:

    return sum(values) / len(values) if values else None





def percentile(values: list[float], p: float) -> float | None:

    if not values:

        return None

    ordered = sorted(values)

    position = (len(ordered) - 1) * p

    lower = math.floor(position)

    upper = math.ceil(position)

    if lower == upper:

        return ordered[lower]

    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)





def standard_deviation(values: list[float]) -> float | None:

    if len(values) < 2:

        return None

    avg = sum(values) / len(values)

    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))





def build_ensemble_summary(data: dict[str, Any]) -> list[dict[str, Any]]:

    hourly = data.get("hourly", {})

    times = hourly.get("time", [])

    result = []



    for i, timestamp in enumerate(times):

        temps = ensemble_member_values(hourly, "temperature_2m", i)

        rain = ensemble_member_values(hourly, "precipitation", i)

        gusts = ensemble_member_values(hourly, "wind_gusts_10m", i)



        rain_wet_members = sum(1 for value in rain if value >= 0.1)

        rain_probability = (rain_wet_members / len(rain) * 100) if rain else None



        result.append({

            "time": timestamp,

            "members": len(temps),

            "temperature_mean_c": mean(temps),

            "temperature_spread_c": standard_deviation(temps),

            "temperature_p10_c": percentile(temps, 0.10),

            "temperature_p90_c": percentile(temps, 0.90),

            "rain_probability_percent": rain_probability,

            "precipitation_mean_mm": mean(rain),

            "precipitation_p90_mm": percentile(rain, 0.90),

            "wind_gust_mean_kmh": mean(gusts),

            "wind_gust_p90_kmh": percentile(gusts, 0.90),

        })



    return result





def build_model_comparison(base: dict[str, Any], ecmwf: dict[str, Any]) -> dict[str, Any]:

    base_hourly = base.get("hourly", {})

    e_hourly = ecmwf.get("hourly", {})

    base_times = base_hourly.get("time", [])

    e_times = e_hourly.get("time", [])



    base_temp = base_hourly.get("temperature_2m", [])

    e_temp = e_hourly.get("temperature_2m", [])



    comparisons = []

    e_lookup = {t: i for i, t in enumerate(e_times)}



    for i, timestamp in enumerate(base_times[:24]):

        j = e_lookup.get(timestamp)

        if j is None:

            continue

        a = safe_float(base_temp[i]) if i < len(base_temp) else None

        b = safe_float(e_temp[j]) if j < len(e_temp) else None

        if a is None or b is None:

            continue

        comparisons.append({

            "time": timestamp,

            "generic_temperature_c": a,

            "ecmwf_temperature_c": b,

            "temperature_difference_c": round(b - a, 2),

        })



    return comparisons





def calculate_confidence(

    model_comparison: list[dict[str, Any]],

    ensemble: list[dict[str, Any]],

) -> dict[str, Any]:

    differences = [abs(x["temperature_difference_c"]) for x in model_comparison]

    spreads = [x["temperature_spread_c"] for x in ensemble if x.get("temperature_spread_c") is not None]



    avg_model_difference = mean(differences)

    avg_spread = mean(spreads)



    score = 90.0

    reasons = []



    if avg_model_difference is not None:

        if avg_model_difference > 3:

            score -= 30

            reasons.append("Forecast models differ noticeably")

        elif avg_model_difference > 1.5:

            score -= 15

            reasons.append("Some model disagreement")

        else:

            reasons.append("Models are broadly aligned")



    if avg_spread is not None:

        if avg_spread > 3:

            score -= 25

            reasons.append("Ensemble spread is high")

        elif avg_spread > 1.5:

            score -= 10

            reasons.append("Moderate ensemble uncertainty")

        else:

            reasons.append("Ensemble spread is relatively low")



    score = max(0, min(100, round(score)))



    if score >= 75:

        label = "HIGH"

    elif score >= 50:

        label = "MEDIUM"

    else:

        label = "LOW"



    return {

        "score": score,

        "level": label,

        "average_model_temperature_difference_c": round(avg_model_difference, 2) if avg_model_difference is not None else None,

        "average_ensemble_temperature_spread_c": round(avg_spread, 2) if avg_spread is not None else None,

        "reasons": reasons,

    }





async def build_weather_intelligence(

    latitude: float,

    longitude: float,

    name: str | None = None,

) -> dict[str, Any]:

    validate_coordinates(latitude, longitude)



    cache_key = f"{round(latitude, 4)}:{round(longitude, 4)}"

    cached = intelligence_cache.get(cache_key)

    if cached and time.time() - cached["timestamp"] < INTELLIGENCE_CACHE_SECONDS:

        return cached["data"]



    base_task = fetch_weather_coordinates(latitude, longitude, name=name)

    ecmwf_task = fetch_ecmwf(latitude, longitude)

    ensemble_task = fetch_ensemble(latitude, longitude)



    base, ecmwf_result, ensemble_raw = await asyncio.gather(

        base_task,

        ecmwf_task,

        ensemble_task,

    )



    ensemble_summary = build_ensemble_summary(ensemble_raw)

    model_comparison = build_model_comparison(base, ecmwf_result)

    confidence = calculate_confidence(model_comparison, ensemble_summary)



    next_hours = ensemble_summary[:24]

    rain_peak = max(

        [x["rain_probability_percent"] for x in next_hours if x.get("rain_probability_percent") is not None],

        default=None,

    )

    gust_peak = max(

        [x["wind_gust_p90_kmh"] for x in next_hours if x.get("wind_gust_p90_kmh") is not None],

        default=None,

    )



    intelligence = {

        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "location": base["location"],

        "summary": {

            "confidence": confidence,

            "next_24h_peak_ensemble_rain_probability_percent": round(rain_peak, 1) if rain_peak is not None else None,

            "next_24h_peak_p90_wind_gust_kmh": round(gust_peak, 1) if gust_peak is not None else None,

        },

        "models": {

            "forecast": "Open-Meteo Best Match",

            "deterministic": "ECMWF IFS HRES",

            "ensemble": "ECMWF IFS 0.25° Ensemble",

        },

        "model_comparison": model_comparison,

        "ensemble": ensemble_summary,

        "method": {

            "type": "model-agreement-and-ensemble-uncertainty",

            "synthetic_data": False,

            "note": "Confidence is an analytical indicator based on model disagreement and ensemble spread; it is not a guarantee of forecast accuracy.",

        },

    }



    intelligence_cache[cache_key] = {"timestamp": time.time(), "data": intelligence}

    return intelligence



# ============================================================

# DEFAULT LOCATIONS SNAPSHOT

# ============================================================



async def build_weather_snapshot() -> dict[str, Any]:

    tasks = [fetch_location(location_id, location) for location_id, location in LOCATIONS.items()]

    results = await asyncio.gather(*tasks, return_exceptions=True)



    locations = []

    for (location_id, location), result in zip(LOCATIONS.items(), results):

        if isinstance(result, Exception):

            locations.append({

                "id": location_id,

                "location": location,

                "error": True,

                "error_message": str(result),

                "source": {"provider": "Open-Meteo", "synthetic_data": False},

            })

        else:

            locations.append(result)



    successful = sum(1 for item in locations if not item.get("error"))

    return {

        "success": True,

        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "source": {"provider": "Open-Meteo", "synthetic_data": False},

        "locations": locations,

        "summary": {

            "total_locations": len(locations),

            "successful_locations": successful,

            "failed_locations": len(locations) - successful,

        },

    }





async def get_cached_weather(force_refresh: bool = False) -> dict[str, Any]:

    current_time = time.time()

    if not force_refresh and cache["data"] is not None and current_time - cache["timestamp"] < CACHE_SECONDS:

        return cache["data"]



    async with cache_lock:

        current_time = time.time()

        if not force_refresh and cache["data"] is not None and current_time - cache["timestamp"] < CACHE_SECONDS:

            return cache["data"]



        data = await build_weather_snapshot()

        cache["data"] = data

        cache["timestamp"] = time.time()

        return data



# ============================================================

# ============================================================
# PHASE 7 — FORECAST INTELLIGENCE
# ============================================================


def build_forecast_response(weather: dict[str, Any]) -> dict[str, Any]:
    """Return a frontend-friendly 24-hour + 7-day forecast payload."""

    hourly = weather.get("hourly", [])
    daily = weather.get("daily", {})

    times = daily.get("time", [])
    codes = daily.get("weather_code", [])
    max_t = daily.get("temperature_max_c", [])
    min_t = daily.get("temperature_min_c", [])
    precip = daily.get("precipitation_sum_mm", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    gust = daily.get("wind_gust_max_kmh", [])

    daily_forecast = []

    for i, day in enumerate(times[:7]):
        code = codes[i] if i < len(codes) else None

        daily_forecast.append({
            "date": day,
            "temperature_max_c": max_t[i] if i < len(max_t) else None,
            "temperature_min_c": min_t[i] if i < len(min_t) else None,
            "precipitation_sum_mm": precip[i] if i < len(precip) else None,
            "precipitation_probability_max_percent": (
                rain_prob[i] if i < len(rain_prob) else None
            ),
            "wind_gust_max_kmh": gust[i] if i < len(gust) else None,
            "weather_code": code,
            "condition": weather_description(code),
        })

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "location": weather.get("location"),
        "timezone": weather.get("observation", {}).get("timezone"),
        "source": {
            "provider": "Open-Meteo",
            "synthetic_data": False,
        },
        "current": weather.get("current"),
        "hourly": hourly[:24],
        "daily": daily_forecast,
        "method": {
            "type": "forecast-aggregation",
            "synthetic_data": False,
            "note": (
                "Forecast values are sourced from Open-Meteo and reformatted "
                "for Zephyr AI visualization."
            ),
        },
    }


@app.get("/api/forecast")
async def get_forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    name: str | None = Query(default=None, max_length=120),
    city: str | None = Query(default=None, max_length=120),
    state: str | None = Query(default=None, max_length=120),
):
    """Return the next 24 hours and next 7 days for Zephyr AI charts."""

    try:
        weather = await fetch_weather_coordinates(
            lat, lon, name=name, city=city, state=state
        )
        return build_forecast_response(weather)

    except httpx.HTTPStatusError as error:
        detail = error.response.text[:500] if error.response is not None else str(error)
        raise HTTPException(status_code=502, detail=f"Forecast provider failed: {detail}")

    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Forecast weather request failed: {error}")

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# API ROUTES

# ============================================================



@app.get("/")

async def root():

    return {

        "name": "Zephyr AI Weather Intelligence API",

        "status": "online",

        "version": "3.1.0",

        "source": "Open-Meteo",

        "synthetic_data": False,

        "intelligence": ["ECMWF IFS HRES", "ECMWF IFS Ensemble"],

        "endpoints": {

            "health": "/health",

            "all_weather": "/api/weather",

            "single_location": "/api/weather/{location_id}",

            "coordinates": "/api/weather/coordinates?lat=28.6139&lon=77.2090",

            "search": "/api/search?q=Delhi",

            "intelligence": "/api/intelligence?lat=28.6139&lon=77.2090",

            "forecast": "/api/forecast?lat=28.6139&lon=77.2090",

            "locations": "/api/locations",

        },

    }





@app.get("/health")

async def health():

    return {

        "status": "healthy",

        "backend": "online",

        "weather_api": "Open-Meteo",

        "ecmwf": "enabled",

        "ensemble": "enabled",

        "synthetic_data": False,

    }





@app.get("/api/locations")

async def get_locations():

    return {"success": True, "locations": LOCATIONS}





@app.get("/api/search")

async def search(q: str = Query(..., min_length=2, max_length=80)):

    try:

        return {"success": True, "query": q, "results": await search_locations(q)}

    except httpx.HTTPError as error:

        raise HTTPException(status_code=502, detail=f"Geocoding API request failed: {error}")





@app.get("/api/weather")

async def get_weather(refresh: bool = False):

    try:

        return await get_cached_weather(force_refresh=refresh)

    except httpx.HTTPError as error:

        raise HTTPException(status_code=502, detail=f"Weather API request failed: {error}")

    except Exception as error:

        raise HTTPException(status_code=500, detail=str(error))





@app.get("/api/weather/coordinates")

async def get_weather_by_coordinates(

    lat: float = Query(..., ge=-90, le=90),

    lon: float = Query(..., ge=-180, le=180),

    name: str | None = Query(default=None, max_length=120),

    city: str | None = Query(default=None, max_length=120),

    state: str | None = Query(default=None, max_length=120),

):

    try:

        return await fetch_weather_coordinates(lat, lon, name=name, city=city, state=state)

    except httpx.HTTPError as error:

        raise HTTPException(status_code=502, detail=f"Weather API request failed: {error}")

    except Exception as error:

        raise HTTPException(status_code=500, detail=str(error))





@app.get("/api/intelligence")

async def get_intelligence(

    lat: float = Query(..., ge=-90, le=90),

    lon: float = Query(..., ge=-180, le=180),

    name: str | None = Query(default=None, max_length=120),

):

    try:

        return await build_weather_intelligence(lat, lon, name=name)

    except httpx.HTTPStatusError as error:

        detail = error.response.text[:500] if error.response is not None else str(error)

        raise HTTPException(status_code=502, detail=f"Weather intelligence provider failed: {detail}")

    except httpx.HTTPError as error:

        raise HTTPException(status_code=502, detail=f"Weather intelligence request failed: {error}")

    except Exception as error:

        raise HTTPException(status_code=500, detail=str(error))





@app.get("/api/weather/{location_id}")

async def get_single_weather(location_id: str):

    location_id = location_id.lower()

    if location_id not in LOCATIONS:

        raise HTTPException(

            status_code=404,

            detail=f"Unknown location '{location_id}'. Available locations: {', '.join(LOCATIONS.keys())}",

        )



    try:

        return await fetch_location(location_id, LOCATIONS[location_id])

    except httpx.HTTPError as error:

        raise HTTPException(status_code=502, detail=f"Weather API request failed: {error}")

    except Exception as error:

        raise HTTPException(status_code=500, detail=str(error))









# ============================================================

# PHASE 8 — ANALYTICAL RISK ENGINE

# ============================================================



def risk_score_for_hour(row: dict[str, Any]) -> tuple[int, list[str]]:

    score = 0

    reasons: list[str] = []

    rain_prob = safe_float(row.get("precipitation_probability")) or 0

    precip = safe_float(row.get("precipitation_mm")) or 0

    gust = safe_float(row.get("wind_gust_kmh")) or 0

    temp = safe_float(row.get("temperature_c")) or 0



    if rain_prob >= 80: score += 35; reasons.append("Very high probability of precipitation")

    elif rain_prob >= 60: score += 25; reasons.append("High probability of precipitation")

    elif rain_prob >= 40: score += 15; reasons.append("Elevated probability of precipitation")

    if precip >= 10: score += 25; reasons.append("Significant precipitation amount possible")

    elif precip >= 5: score += 15; reasons.append("Moderate precipitation amount possible")

    elif precip >= 2: score += 8; reasons.append("Measurable precipitation possible")

    if gust >= 70: score += 35; reasons.append("Very strong wind gusts possible")

    elif gust >= 50: score += 25; reasons.append("Strong wind gusts possible")

    elif gust >= 35: score += 15; reasons.append("Elevated wind gusts possible")

    if temp >= 45: score += 35; reasons.append("Extreme heat range")

    elif temp >= 40: score += 25; reasons.append("Very high temperature range")

    elif temp >= 37: score += 15; reasons.append("High temperature range")

    code = row.get("weather_code")

    if code in [95,96,99]: score += 35; reasons.append("Thunderstorm signal")

    return min(100, score), reasons





def risk_level(score: int) -> str:

    return "HIGH" if score >= 70 else "MODERATE" if score >= 40 else "LOW"





@app.get("/api/risk")

async def get_risk(

    lat: float = Query(..., ge=-90, le=90),

    lon: float = Query(..., ge=-180, le=180),

    name: str | None = Query(default=None, max_length=120),

):

    try:

        data = await fetch_weather_coordinates(lat, lon, name=name)

        timeline = []

        for row in data.get("hourly", [])[:24]:

            score, reasons = risk_score_for_hour(row)

            timeline.append({"time": row.get("time"), "score": score, "level": risk_level(score), "reasons": reasons})

        if not timeline:

            timeline = [{"time": None, "score": 0, "level": "LOW", "reasons": []}]

        critical = max(timeline, key=lambda x: x["score"])

        return {

            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

            "location": data.get("location"),

            "overall": {"score": critical["score"], "level": critical["level"]},

            "critical_window": critical,

            "timeline": timeline,

            "method": {

                "synthetic_data": False,

                "type": "forecast-variable risk analysis",

                "note": "Risk levels are analytical indicators derived from forecast variables. They are not official weather warnings."

            }

        }

    except httpx.HTTPError as error:

        raise HTTPException(status_code=502, detail=f"Risk weather provider failed: {error}")

    except Exception as error:

        raise HTTPException(status_code=500, detail=str(error))





# ============================================================

# PHASE 8 — GROUNDED AI WEATHER ANALYST

# ============================================================



class AIAnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    name: str | None = Field(default=None, max_length=120)
    api_key: str | None = Field(default=None, max_length=1000)

class AIModelsRequest(BaseModel):
    api_key: str = Field(..., min_length=10, max_length=1000)

async def detect_nvidia_models(api_key: str) -> list[dict[str, Any]]:
    if http_client is None:
        raise RuntimeError("HTTP client is not initialized.")
    response = await http_client.get(NVIDIA_MODELS_URL, headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
    response.raise_for_status()
    models = response.json().get("data") or []
    return [{"id": x.get("id"), "owned_by": x.get("owned_by"), "object": x.get("object", "model")} for x in models if x.get("id")]

async def resolve_nvidia_model(api_key: str) -> str:
    models = await detect_nvidia_models(api_key)
    ids = [x["id"] for x in models]
    if NVIDIA_MODEL and NVIDIA_MODEL in ids:
        return NVIDIA_MODEL
    for preferred in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "meta/llama-3.1-8b-instruct"]:
        if preferred in ids:
            return preferred
    if not ids:
        raise RuntimeError("NVIDIA API returned no available models.")
    return ids[0]

async def call_weather_ai(question: str, context: dict[str, Any], api_key: str) -> tuple[str, str]:
    if not api_key:
        raise HTTPException(status_code=503, detail="No AI API key supplied. Add your NVIDIA API key in Ask Zephyr or configure backend/.env.")
    model = await resolve_nvidia_model(api_key)
    system_prompt = ("You are Zephyr, a weather intelligence analyst. Answer using ONLY the supplied weather context. "
        "Do not invent observations, warnings, locations, or numbers. Clearly distinguish forecast probability from certainty. "
        "Be concise and practical. If the question asks whether something is safe, phrase the answer as forecast-based information, "
        "not an official safety warning. Mention the relevant time window and uncertainty when useful.")
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Question: {question}\n\nWeather context (JSON):\n{context}"}], "temperature": 0.2, "max_tokens": 700, "stream": False}
    response = await http_client.post(NVIDIA_API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}, json=payload)
    response.raise_for_status()
    choices = response.json().get("choices") or []
    if not choices:
        raise RuntimeError("AI provider returned no choices.")
    content = choices[0].get("message", {}).get("content")
    if not content:
        raise RuntimeError("AI provider returned an empty answer.")
    return str(content).strip(), model

@app.post("/api/ai/models")
async def ai_models(request: AIModelsRequest):
    try:
        models = await detect_nvidia_models(request.api_key.strip())
        return {"provider": "NVIDIA", "models": models, "detected_model": models[0]["id"] if models else None}
    except httpx.HTTPStatusError as error:
        detail = error.response.text[:300] if error.response is not None else str(error)
        status = 401 if error.response is not None and error.response.status_code in (401, 403) else 502
        raise HTTPException(status_code=status, detail=f"NVIDIA model discovery failed: {detail}")
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"NVIDIA model discovery request failed: {error}")
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/api/ai/analyze")
async def ai_analyze(request: AIAnalyzeRequest):
    try:
        weather = await fetch_weather_coordinates(request.latitude, request.longitude, name=request.name)
        intelligence = await build_weather_intelligence(request.latitude, request.longitude, name=request.name)
        risk = await get_risk(request.latitude, request.longitude, name=request.name)
        context = {"location": weather.get("location"), "current": weather.get("current"), "next_24_hours": weather.get("hourly", [])[:24], "daily_forecast": weather.get("daily"), "forecast_intelligence_summary": intelligence.get("summary"), "model_comparison": intelligence.get("model_comparison", [])[:8], "risk": risk, "data_sources": intelligence.get("models")}
        answer, model = await call_weather_ai(request.question, context, (request.api_key or NVIDIA_API_KEY or "").strip())
        return {"answer": answer, "model": model, "location": weather.get("location"), "grounded": True, "synthetic_data": False, "sources": ["Open-Meteo", "ECMWF", "ECMWF ensemble"]}
    except HTTPException:
        raise
    except httpx.HTTPStatusError as error:
        detail = error.response.text[:500] if error.response is not None else str(error)
        status = 401 if error.response is not None and error.response.status_code in (401, 403) else 502
        raise HTTPException(status_code=status, detail=f"AI provider failed: {detail}")
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"AI provider request failed: {error}")
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# RUN

# ============================================================



if __name__ == "__main__":

    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
