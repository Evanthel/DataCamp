
from datetime import datetime, timedelta


def format_date(timestamp, datetime_format):
    dt_object = (datetime.fromtimestamp(timestamp)).strftime(datetime_format)
    return dt_object


def calculate_landing_time(rocket_launch_dt, travel_duration):
    landing_date_string = (timedelta(days=travel_duration) +
                           rocket_launch_dt).strftime("%d-%m-%Y")
    return landing_date_string


def days_until_delivery(expected_delivery_dt, current_dt):
    days_until = (expected_delivery_dt - current_dt).days
    return days_until
