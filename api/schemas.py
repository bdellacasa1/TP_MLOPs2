from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    gender: Literal[
        "Female",
        "Male",
    ] = Field(alias="Gender")

    customer_type: Literal[
        "Loyal Customer",
        "disloyal Customer",
    ] = Field(alias="Customer Type")

    type_of_travel: Literal[
        "Business travel",
        "Personal Travel",
    ] = Field(alias="Type of Travel")

    travel_class: Literal[
        "Business",
        "Eco",
        "Eco Plus",
    ] = Field(alias="Class")

    age: int = Field(
        alias="Age",
        ge=0,
        le=120,
    )

    flight_distance: int = Field(
        alias="Flight Distance",
        ge=0,
    )

    departure_delay: float = Field(
        alias="Departure Delay in Minutes",
        ge=0,
    )

    arrival_delay: float = Field(
        alias="Arrival Delay in Minutes",
        ge=0,
    )


class PredictionResponse(BaseModel):

    prediction: int
    label: str
    probability: float

    model_name: str
    model_alias: str
    model_version: str