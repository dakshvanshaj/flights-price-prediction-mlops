from datetime import date
from enum import Enum
from pydantic import BaseModel, StrictFloat


class LocationEnum(str, Enum):
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
    ECONOMIC = "economic"
    FIRST_CLASS = "firstClass"
    PREMIUM = "premium"


class AgencyEnum(str, Enum):
    CLOUDFY = "CloudFy"
    FLYINGDROPS = "FlyingDrops"
    RAINBOW = "Rainbow"


class FlightInput(BaseModel):
    """
    Represents the raw features for a single flight prediction request.
    Pydantic will enforce data types and allowed values.
    """

    from_location: LocationEnum
    to_location: LocationEnum
    flight_type: FlightTypeEnum
    time: int
    distance: int
    agency: AgencyEnum
    date: date


class PredictionOutput(BaseModel):
    """
    Represents the output of a prediction request.
    """

    predicted_price: StrictFloat
