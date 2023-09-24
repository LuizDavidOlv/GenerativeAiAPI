from pydantic import BaseModel
#from .Enums.PineconeMetricEnum import MetricEnum

class CreateIndexModel(BaseModel):
    index_name: str
    dimension: int
    metric: str
