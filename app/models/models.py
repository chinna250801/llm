from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class InputParam(BaseModel):
    product_id: str
    granularity: str = "ONE_DAY"
    limit: int = 350
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
