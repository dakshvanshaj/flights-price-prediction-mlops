"""
Pydantic Schemas for API Data Validation.

This module defines the data structures for API requests and responses, ensuring
that all incoming and outgoing data conforms to a predefined contract.
It uses Pydantic for robust data validation, conversion, and documentation.
"""

from datetime import date
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class LocationEnum(str, Enum):
    """Allowed locations for flights."""

    ARACAJU_SE = "Aracaju (SE)"
    BRASILIA_DF = "Brasilia (DF)"
    CAMPO_GRANDE_MS = "Campo Grande (MS)"
    FLORIANOPOLIS_SC = "Florianopolis (SC)"
    NATAL_RN = "Natal (RN)"
    RECIFE_PE = "Recife (PE)"
    RIO_DE_JANEIRO_RJ = "Rio de Janeiro (RJ)"
    SALVADOR_BH = "Salvador (BH)"
    SAO_PAULO_SP = "Sao Paulo (SP)"


class FlightTypeEnum(str, Enum):
    """Allowed flight types."""

    ECONOMIC = "economic"
    FIRST_CLASS = "firstClass"
    PREMIUM = "premium"


class AgencyEnum(str, Enum):
    """Allowed travel agencies."""

    CLOUDFY = "CloudFy"
    FLYINGDROPS = "FlyingDrops"
    RAINBOW = "Rainbow"


class InputSchema(BaseModel):
    """
    Represents the raw features for a single flight prediction request.
    Pydantic enforces data types, constraints, and allowed values.
    """

    from_location: LocationEnum
    to_location: LocationEnum
    flight_type: FlightTypeEnum
    time: float = Field(..., gt=0, description="Duration of the flight in hours.")
    distance: float = Field(
        ..., gt=0, description="Distance of the flight in kilometers."
    )
    agency: AgencyEnum
    date: date

    # NEW: Add an example for the auto-generated API documentation
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "from_location": "Recife (PE)",
                "to_location": "Florianopolis (SC)",
                "flight_type": "firstClass",
                "time": 1.76,
                "distance": 676.53,
                "agency": "FlyingDrops",
                "date": "2019-09-26",
            }
        }
    )


class OutputSchema(BaseModel):
    """Represents the output of a successful prediction request."""

    predicted_price: float = Field(
        ..., description="The predicted price of the flight."
    )
