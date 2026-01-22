"""
Unit conversion tool for temperature, length, and weight.
"""

from langchain_core.tools import tool

from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert between different units of measurement.
    
    Supported conversions:
    - Temperature: celsius, fahrenheit, kelvin
    - Length: meter, kilometer, mile, foot, inch
    - Weight: kilogram, gram, pound, ounce
    
    Parameters
    ----------
    value : float
        Value to convert
    from_unit : str
        Source unit (e.g., 'celsius', 'meter', 'kilogram')
    to_unit : str
        Target unit (e.g., 'fahrenheit', 'kilometer', 'pound')
        
    Returns
    -------
    dict
        Conversion result with value and units
        
    Examples
    --------
    >>> unit_converter(100, "celsius", "fahrenheit")
    {'status': 'success', 'value': 100, 'from_unit': 'celsius', 'to_unit': 'fahrenheit', 'result': 212.0}
    """
    try:
        logger.info(f"Converting {value} from {from_unit} to {to_unit}")
        
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # Temperature conversions
        temp_units = {"celsius", "fahrenheit", "kelvin"}

        # Length conversions
        length_units = {"meter", "kilometer", "mile", "foot", "inch"}

        # Weight conversions
        weight_units = {"kilogram", "gram", "pound", "ounce"}

        if from_unit in temp_units and to_unit in temp_units:
            result = _convert_temperature(value, from_unit, to_unit)
        
        elif from_unit in length_units and to_unit in length_units:
            result = _convert_length(value, from_unit, to_unit)
        
        elif from_unit in weight_units and to_unit in weight_units:
            result = _convert_weight(value, from_unit, to_unit)
        
        else:
            logger.warning(f"Unsupported conversion: {from_unit} to {to_unit}")
            return {
                "status": "error",
                "error_type": "unsupported_conversion",
                "message": f"Unsupported conversion from {from_unit} to {to_unit}. "
                          f"Supported: temperature (celsius/fahrenheit/kelvin), "
                          f"length (meter/kilometer/mile/foot/inch), "
                          f"weight (kilogram/gram/pound/ounce)"
            }
        
        logger.info(f"Conversion result: {result}")
        return {
            "status": "success",
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": round(result, 4)
        }
        
    except Exception as e:
        logger.error(f"Unit converter error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_type": "unexpected_error",
            "message": str(e)
        }


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between celsius, fahrenheit, and kelvin."""
    # Convert to Celsius first
    if from_unit == "fahrenheit":
        celsius = (value - 32) * 5/9
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        celsius = value
    
    # Convert from Celsius to target
    if to_unit == "fahrenheit":
        return celsius * 9/5 + 32
    elif to_unit == "kelvin":
        return celsius + 273.15
    else:
        return celsius


def _convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """Convert length between meter, kilometer, mile, foot, and inch."""
    # Convert to meters first
    to_meters = {
        "meter": 1,
        "kilometer": 1000,
        "mile": 1609.34,
        "foot": 0.3048,
        "inch": 0.0254
    }
    
    meters = value * to_meters[from_unit]
    return meters / to_meters[to_unit]


def _convert_weight(value: float, from_unit: str, to_unit: str) -> float:
    """Convert weight between kilogram, gram, pound, and ounce."""
    # Convert to kilograms first
    to_kilograms = {
        "kilogram": 1,
        "gram": 0.001,
        "pound": 0.453592,
        "ounce": 0.0283495
    }
    
    kilograms = value * to_kilograms[from_unit]
    return kilograms / to_kilograms[to_unit]