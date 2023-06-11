import pandas as pd
import re


class DataFrameFilterException(Exception):
    pass


def generate_dataframe_filter(input_str, field_name="Register Number"):

    filters = input_str.split(',')

    filter_condition = None
    filter_list = []
    current_filter = ""
    for f in filters:
        if not f:
            return None

        if '..' in f:
            try:
                start, end = map(int, f.split('..'))
                if start > end:
                    raise DataFrameFilterException(
                        f"Invalid range: {f}. Start value cannot be greater than end value.")
                current_filter = f"`{field_name}` >= {start} & `{field_name}` <= {end}"
            except ValueError:
                raise DataFrameFilterException(
                    f"Invalid input format: {f}. Please provide input in the format 'start..end' or an individual number.")
        else:
            try:
                value = int(f)
                current_filter = f"`{field_name}` == {value}"
            except ValueError:
                raise DataFrameFilterException(
                    f"Invalid input format: {f}. Please provide input in the format 'start..end' or an individual number.")

        filter_list.append(current_filter)
    filter_condition = " | ".join(filter_list)
    return filter_condition
