# accept the input params/data payload
# output the score, class, latency

from pydantic import BaseModel
from pydantic import EmailStr, HttpUrl 

class DataInput(BaseModel):
    video_url: list[HttpUrl]






