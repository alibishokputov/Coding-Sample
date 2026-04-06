from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Optional

import numpy as np
import pandas as pd
import googlemaps

api_key = 'AIzaSyCQSEIhJtmkevRygoAO3Dfy-22pP188CWI'


@dataclass
class CommuteAPIScenarioConfig:
    reference_datetime: datetime = field(
        default_factory=lambda: datetime.now() + timedelta(minutes=5)
    )

    fallback_day: str = "wednesday"
    fallback_time_window: str = "morning_06_10"
    fallback_mode: str = "Drive alone"

    window_to_time: dict = None

    canonical_day_order: list = None


    min_duration_minutes: float = 1.0
    max_duration_minutes: float = 300.0
    topcode_duration_minutes: float = 180.0
    min_distance_miles: float = 0.05
    max_distance_miles: float = 300.0

    def __post_init__(self):
        if self.window_to_time is None:
            self.window_to_time = {
                "early_morning_00_06": time(5, 0),
                "morning_06_10": time(8, 0),
                "midday_10_14": time(12, 0),
                "afternoon_14_18": time(16, 0),
                "evening_18_21": time(19, 0),
                "late_night_21_00": time(22, 0),
            }

        if self.canonical_day_order is None:
            self.canonical_day_order = [
                "wednesday", "tuesday", "thursday", "monday", "friday", "saturday", "sunday"
            ]


DAY_TO_NUM = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

def _next_occurrence(reference_dt: datetime, target_weekday: int) -> datetime.date:
    days_ahead = (target_weekday - reference_dt.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (reference_dt + timedelta(days=days_ahead)).date()


def identify_canonical_commute_day(
    row: pd.Series,
    day_order: Optional[list[str]] = None,
    suffix: str = "_to_work_time",
) -> Optional[str]:
    """
    Identify a canonical observed commute day from day-specific to-work window fields.
    """
    if day_order is None:
        day_order = ["wednesday", "tuesday", "thursday", "monday", "friday", "saturday", "sunday"]

    for day in day_order:
        col = f"{day}{suffix}"
        if col in row.index and pd.notna(row[col]):
            return day
    return None


def resolve_api_mode(mode_value: Optional[str], fallback_mode: str = "Drive alone") -> dict:
    """
    Map collapsed reported mode into Google Directions API parameters.
    """
    mode_value = fallback_mode if pd.isna(mode_value) or mode_value is None else mode_value

    if mode_value in {"Drive alone", "Shared ride"}:
        return {
            "google_mode": "driving",
            "transit_mode": None,
            "scenario_mode_source": "reported" if mode_value != fallback_mode else "fallback",
            "mode_collapsed_used": mode_value,
        }

    if mode_value == "Bus":
        return {
            "google_mode": "transit",
            "transit_mode": "bus",
            "scenario_mode_source": "reported" if mode_value != fallback_mode else "fallback",
            "mode_collapsed_used": mode_value,
        }

    if mode_value == "Rail":
        return {
            "google_mode": "transit",
            "transit_mode": "rail",
            "scenario_mode_source": "reported" if mode_value != fallback_mode else "fallback",
            "mode_collapsed_used": mode_value,
        }

    if mode_value == "Bike":
        return {
            "google_mode": "bicycling",
            "transit_mode": None,
            "scenario_mode_source": "reported" if mode_value != fallback_mode else "fallback",
            "mode_collapsed_used": mode_value,
        }

    if mode_value == "Walk":
        return {
            "google_mode": "walking",
            "transit_mode": None,
            "scenario_mode_source": "reported" if mode_value != fallback_mode else "fallback",
            "mode_collapsed_used": mode_value,
        }

    return {
        "google_mode": "driving",
        "transit_mode": None,
        "scenario_mode_source": "fallback_unknown_mode",
        "mode_collapsed_used": fallback_mode,
    }


def resolve_departure_datetime(
    row: pd.Series,
    config: CommuteAPIScenarioConfig,
    primary_time_window_col: str = "primary_time_window",
) -> tuple[Optional[datetime], str, str, str]:
    """
    Build respondent-specific departure datetime.
    """
    primary_day = identify_canonical_commute_day(
        row,
        day_order=config.canonical_day_order,
        suffix="_to_work_time",
    )
    day_used = primary_day if primary_day is not None else config.fallback_day

    primary_window = row.get(primary_time_window_col, np.nan)
    if pd.isna(primary_window):
        time_window_used = config.fallback_time_window
    else:
        time_window_used = primary_window

    dep_time = config.window_to_time.get(
        time_window_used,
        config.window_to_time[config.fallback_time_window]
    )

    dep_date = _next_occurrence(config.reference_datetime, DAY_TO_NUM[day_used])
    departure_dt = datetime.combine(dep_date, dep_time)

    if primary_day is None and pd.isna(primary_window):
        scenario_time_source = "fallback_day_and_time"
    elif primary_day is None:
        scenario_time_source = "fallback_day_reported_time"
    elif pd.isna(primary_window):
        scenario_time_source = "reported_day_fallback_time"
    else:
        scenario_time_source = "reported_day_and_time"

    return departure_dt, day_used, time_window_used, scenario_time_source


def extract_transit_details(directions_result):
    out = {
        "api_n_steps": np.nan,
        "api_n_transit_segments": 0,
        "api_n_transfers": 0,
        "api_n_walking_segments": 0,
        "api_has_walking_segment": 0,
        "api_transit_vehicle_types": pd.NA,
        "api_transit_line_short_names": pd.NA,
        "api_transit_line_names": pd.NA,
        "api_transit_headsigns": pd.NA,
        "api_transit_agencies": pd.NA,
        "api_transit_num_stops_total": np.nan,
        "api_transit_departure_stop_names": pd.NA,
        "api_transit_arrival_stop_names": pd.NA,
        "api_transit_departure_times": pd.NA,
        "api_transit_arrival_times": pd.NA,
    }

    if not directions_result:
        return out

    try:
        leg = directions_result[0]["legs"][0]
        steps = leg.get("steps", [])
        out["api_n_steps"] = len(steps)

        vehicle_types = []
        line_short_names = []
        line_names = []
        headsigns = []
        agencies = []
        dep_stops = []
        arr_stops = []
        dep_times = []
        arr_times = []
        num_stops_total = 0

        for step in steps:
            travel_mode = step.get("travel_mode")

            if travel_mode == "WALKING":
                out["api_n_walking_segments"] += 1
                out["api_has_walking_segment"] = 1

            if travel_mode == "TRANSIT":
                out["api_n_transit_segments"] += 1
                td = step.get("transit_details", {})
                line = td.get("line", {})
                vehicle = line.get("vehicle", {})

                if vehicle.get("type"):
                    vehicle_types.append(vehicle["type"])
                if line.get("short_name"):
                    line_short_names.append(line["short_name"])
                if line.get("name"):
                    line_names.append(line["name"])
                if td.get("headsign"):
                    headsigns.append(td["headsign"])

                for agency in line.get("agencies", []):
                    if agency.get("name"):
                        agencies.append(agency["name"])

                dep_stop = td.get("departure_stop", {}).get("name")
                arr_stop = td.get("arrival_stop", {}).get("name")
                if dep_stop:
                    dep_stops.append(dep_stop)
                if arr_stop:
                    arr_stops.append(arr_stop)

                dep_time = td.get("departure_time", {}).get("text")
                arr_time = td.get("arrival_time", {}).get("text")
                if dep_time:
                    dep_times.append(dep_time)
                if arr_time:
                    arr_times.append(arr_time)

                num_stops_total += td.get("num_stops", 0)

        out["api_n_transfers"] = max(out["api_n_transit_segments"] - 1, 0)

        if vehicle_types:
            out["api_transit_vehicle_types"] = "|".join(pd.unique(pd.Series(vehicle_types)).tolist())
        if line_short_names:
            out["api_transit_line_short_names"] = "|".join(pd.unique(pd.Series(line_short_names)).tolist())
        if line_names:
            out["api_transit_line_names"] = "|".join(pd.unique(pd.Series(line_names)).tolist())
        if headsigns:
            out["api_transit_headsigns"] = "|".join(headsigns)
        if agencies:
            out["api_transit_agencies"] = "|".join(pd.unique(pd.Series(agencies)).tolist())
        if dep_stops:
            out["api_transit_departure_stop_names"] = "|".join(dep_stops)
        if arr_stops:
            out["api_transit_arrival_stop_names"] = "|".join(arr_stops)
        if dep_times:
            out["api_transit_departure_times"] = "|".join(dep_times)
        if arr_times:
            out["api_transit_arrival_times"] = "|".join(arr_times)

        out["api_transit_num_stops_total"] = (
            num_stops_total if out["api_n_transit_segments"] > 0 else np.nan
        )

        return out

    except Exception:
        return out


def clean_api_outputs(
    distance_miles: float,
    duration_minutes: float,
    config: CommuteAPIScenarioConfig
) -> dict:
    """
    Apply explicit cleaning and flagging to API outputs.
    """
    distance = pd.to_numeric(pd.Series([distance_miles]), errors="coerce").iloc[0]
    duration = pd.to_numeric(pd.Series([duration_minutes]), errors="coerce").iloc[0]

    out = {
        "api_distance_miles_raw": distance,
        "api_duration_minutes_raw": duration,
        "api_flag_route_missing": int(pd.isna(distance) or pd.isna(duration)),
        "api_flag_duration_below_min": 0,
        "api_flag_duration_above_max": 0,
        "api_flag_duration_topcoded": 0,
        "api_flag_distance_below_min": 0,
        "api_flag_distance_above_max": 0,
        "api_distance_miles_clean": np.nan,
        "api_duration_minutes_clean": np.nan,
        "api_duration_minutes_topcoded": np.nan,
    }

    if out["api_flag_route_missing"] == 1:
        return out

    if duration < config.min_duration_minutes:
        out["api_flag_duration_below_min"] = 1
        duration = np.nan
    elif duration > config.max_duration_minutes:
        out["api_flag_duration_above_max"] = 1
        duration = np.nan

    if distance < config.min_distance_miles:
        out["api_flag_distance_below_min"] = 1
        distance = np.nan
    elif distance > config.max_distance_miles:
        out["api_flag_distance_above_max"] = 1
        distance = np.nan

    out["api_distance_miles_clean"] = distance
    out["api_duration_minutes_clean"] = duration

    if pd.notna(duration):
        out["api_flag_duration_topcoded"] = int(duration > config.topcode_duration_minutes)
        out["api_duration_minutes_topcoded"] = min(duration, config.topcode_duration_minutes)

    return out


def get_commute_details_scenario(
    gmaps_client,
    origin_lat,
    origin_lon,
    dest_lat,
    dest_lon,
    google_mode,
    departure_dt,
    transit_mode=None,
):
    """
    Query Google Directions API and return raw response plus summary fields.
    """
    origin = (origin_lat, origin_lon)
    destination = (dest_lat, dest_lon)

    try:
        kwargs = {
            "origin": origin,
            "destination": destination,
            "mode": google_mode,
        }

        if google_mode in {"driving", "transit"} and departure_dt is not None:
            kwargs["departure_time"] = departure_dt

        if google_mode == "transit" and transit_mode is not None:
            kwargs["transit_mode"] = transit_mode

        directions_result = gmaps_client.directions(**kwargs)

        if not directions_result:
            return {
                "distance_miles": np.nan,
                "duration_minutes": np.nan,
                "duration_source": "no_route",
                "directions_result": None,
            }

        leg = directions_result[0]["legs"][0]
        distance_meters = leg["distance"]["value"]

        if google_mode == "driving" and "duration_in_traffic" in leg:
            duration_seconds = leg["duration_in_traffic"]["value"]
            duration_source = "duration_in_traffic"
        else:
            duration_seconds = leg["duration"]["value"]
            duration_source = "duration"

        distance_miles = distance_meters * 0.000621371
        duration_minutes = duration_seconds / 60.0

        return {
            "distance_miles": round(distance_miles, 2),
            "duration_minutes": round(duration_minutes, 2),
            "duration_source": duration_source,
            "directions_result": directions_result,
        }

    except googlemaps.exceptions.ApiError as e:
        return {
            "distance_miles": np.nan,
            "duration_minutes": np.nan,
            "duration_source": f"api_error:{e}",
            "directions_result": None,
        }
    except googlemaps.exceptions.TransportError as e:
        return {
            "distance_miles": np.nan,
            "duration_minutes": np.nan,
            "duration_source": f"transport_error:{e}",
            "directions_result": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "distance_miles": np.nan,
            "duration_minutes": np.nan,
            "duration_source": f"request_error:{e}",
            "directions_result": None,
        }
    except Exception as e:
        return {
            "distance_miles": np.nan,
            "duration_minutes": np.nan,
            "duration_source": f"unexpected_error:{type(e).__name__}:{e}",
            "directions_result": None,
        }


def build_api_commute_measure_for_row(
    row: pd.Series,
    gmaps_client: googlemaps.Client,
    config: CommuteAPIScenarioConfig = CommuteAPIScenarioConfig(),
    origin_lat_col: str = "home_lat",
    origin_lon_col: str = "home_lon",
    dest_lat_col: str = "work_lat",
    dest_lon_col: str = "work_lon",
    primary_mode_col: str = "primary_mode_collapsed",
    primary_time_window_col: str = "primary_time_window",
) -> pd.Series:
    """
    Build respondent-level API travel measure using:
    """
    origin_lat = row.get(origin_lat_col)
    origin_lon = row.get(origin_lon_col)
    dest_lat = row.get(dest_lat_col)
    dest_lon = row.get(dest_lon_col)

    has_valid_coords = all(pd.notna([origin_lat, origin_lon, dest_lat, dest_lon]))

    out = {
        "api_has_valid_home_work_coords": int(has_valid_coords),
        "api_mode_used": pd.NA,
        "api_google_mode": pd.NA,
        "api_transit_mode": pd.NA,
        "api_primary_day_used": pd.NA,
        "api_primary_time_window_used": pd.NA,
        "api_departure_datetime_used": pd.NaT,
        "api_scenario_mode_source": pd.NA,
        "api_scenario_time_source": pd.NA,
        "api_duration_measure_source": pd.NA,
        "api_scenario_type": pd.NA,

        # Route summary outputs
        "api_distance_miles_raw": np.nan,
        "api_duration_minutes_raw": np.nan,
        "api_flag_route_missing": 1,
        "api_flag_duration_below_min": 0,
        "api_flag_duration_above_max": 0,
        "api_flag_duration_topcoded": 0,
        "api_flag_distance_below_min": 0,
        "api_flag_distance_above_max": 0,
        "api_distance_miles_clean": np.nan,
        "api_duration_minutes_clean": np.nan,
        "api_duration_minutes_topcoded": np.nan,

        # Transit / step metadata
        "api_n_steps": np.nan,
        "api_n_transit_segments": 0,
        "api_n_walking_segments": 0,
        "api_has_walking_segment": 0,
        "api_transit_vehicle_types": pd.NA,
        "api_transit_line_short_names": pd.NA,
        "api_transit_agencies": pd.NA,
        "api_transit_num_stops_total": np.nan,
        "api_transit_departure_stop_names": pd.NA,
        "api_transit_arrival_stop_names": pd.NA,
    }

    if not has_valid_coords:
        out["api_scenario_type"] = "missing_coordinates"
        return pd.Series(out)

    mode_info = resolve_api_mode(
        row.get(primary_mode_col, np.nan),
        fallback_mode=config.fallback_mode
    )

    departure_dt, day_used, time_window_used, scenario_time_source = resolve_departure_datetime(
        row=row,
        config=config,
        primary_time_window_col=primary_time_window_col,
    )

    has_observed_mode = pd.notna(row.get(primary_mode_col, np.nan))
    has_observed_time = pd.notna(row.get(primary_time_window_col, np.nan))
    has_observed_day = identify_canonical_commute_day(
        row,
        day_order=config.canonical_day_order,
        suffix="_to_work_time",
    ) is not None

    if has_observed_mode and has_observed_time and has_observed_day:
        scenario_type = "observed_mode_day_time"
    elif has_observed_mode or has_observed_time or has_observed_day:
        scenario_type = "partially_observed_with_fallback"
    else:
        scenario_type = "full_fallback_noncommute_or_remote"

    out.update({
        "api_mode_used": mode_info["mode_collapsed_used"],
        "api_google_mode": mode_info["google_mode"],
        "api_transit_mode": mode_info["transit_mode"] if mode_info["transit_mode"] else pd.NA,
        "api_primary_day_used": day_used,
        "api_primary_time_window_used": time_window_used,
        "api_departure_datetime_used": departure_dt,
        "api_scenario_mode_source": mode_info["scenario_mode_source"],
        "api_scenario_time_source": scenario_time_source,
        "api_scenario_type": scenario_type,
    })

    route_result = get_commute_details_scenario(
        gmaps_client=gmaps_client,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        google_mode=mode_info["google_mode"],
        departure_dt=departure_dt,
        transit_mode=mode_info["transit_mode"],
    )

    out["api_duration_measure_source"] = route_result["duration_source"]

    transit_meta = extract_transit_details(route_result["directions_result"])
    out.update(transit_meta)

    cleaned = clean_api_outputs(
        distance_miles=route_result["distance_miles"],
        duration_minutes=route_result["duration_minutes"],
        config=config
    )
    out.update(cleaned)

    return pd.Series(out)



def build_api_commute_measures(
    df: pd.DataFrame,
    gmaps_client: googlemaps.Client,
    config: Optional[CommuteAPIScenarioConfig] = None,
    origin_lat_col: str = "home_lat",
    origin_lon_col: str = "home_lon",
    dest_lat_col: str = "work_lat",
    dest_lon_col: str = "work_lon",
    primary_mode_col: str = "primary_mode_collapsed",
    primary_time_window_col: str = "primary_time_window",
) -> pd.DataFrame:
    """
    Apply respondent-level API measure builder to an entire DataFrame.
    """
    if config is None:
        config = CommuteAPIScenarioConfig()

    return df.apply(
        lambda row: build_api_commute_measure_for_row(
            row=row,
            gmaps_client=gmaps_client,
            config=config,
            origin_lat_col=origin_lat_col,
            origin_lon_col=origin_lon_col,
            dest_lat_col=dest_lat_col,
            dest_lon_col=dest_lon_col,
            primary_mode_col=primary_mode_col,
            primary_time_window_col=primary_time_window_col,
        ),
        axis=1,
    )