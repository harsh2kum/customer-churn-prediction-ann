from pydantic import BaseModel
from enum import Enum

class GeographyEnum(str, Enum):
    France = "France"
    Germany = "Germany"
    Spain = "Spain"

class GenderEnum(str, Enum):
    Male = "Male"
    Female = "Female"

class CustomerData(BaseModel):
    CreditScore: int
    Geography: GeographyEnum
    Gender: GenderEnum
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
