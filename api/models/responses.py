from pydantic import BaseModel
from typing import Dict, List, Optional

class StartResponse(BaseModel):
    status: str
    message: str
    model_name: str
    model_id: int
    streams_started: int
    stream_details: List[Dict]

class StopResponse(BaseModel):
    status: str
    message: str
    model_name: str

class StatusResponse(BaseModel):
    model_name: str
    model_id: int
    status: str
    streams_running: int
    uptime: Optional[float]
    stats: Dict
