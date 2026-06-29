from typing import List, Literal
from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """A concrete task that must be completed after the meeting."""
    owner: str = Field(description="Full name or role of the person responsible")
    task: str = Field(description="Clear description of what must be done")
    deadline: str = Field(description="Deadline if mentioned, otherwise 'not specified'")


class MeetingMinutes(BaseModel):
    """
    Complete structured minutes for one meeting (or one chunk of a meeting).

    This model is passed as `response_format=MeetingMinutes` to create_agent().
    The agent's response is then read from result["structured_response"].
    """
    summary: str = Field(
        description="One concise paragraph summarising the main topics discussed"
    )
    decisions: List[str] = Field(
        description="Key decisions made. Each item is a distinct decision. No duplicates."
    )
    action_items: List[ActionItem] = Field(
        description="Concrete tasks assigned to people, with owner and deadline"
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall tone of the meeting"
    )