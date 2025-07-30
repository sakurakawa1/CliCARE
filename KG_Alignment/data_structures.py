# FILE: data_structures.py
# Description: Defines data models matching the graph schema using Pydantic to ensure data consistency and code clarity

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- Data Models for the Guideline Knowledge Graph (Gg) ---

class GuidelineNode(BaseModel):
    """Base model for a node in the guideline graph"""
    node_id: int
    labels: List[str]
    properties: Dict[str, Any]

class GuidelinePathway:
    """Represents a standardized treatment pathway extracted from a guideline"""
    def __init__(self, pathway_id: str, steps: Optional[List[GuidelineNode]]):
        self.pathway_id = pathway_id
        self.steps = steps if steps is not None else []

    def __repr__(self):
        step_names = [s.properties.get('name', 'Unknown') for s in self.steps]
        return f"Pathway(id={self.pathway_id}, steps={step_names})"


# --- Data Models for the Clinical Temporal Knowledge Graph (Gt) ---

class ClinicalEvent(BaseModel):
    """Represents a clinical event, such as medication, examination, etc."""
    node_id: int
    labels: List[str]
    properties: Dict[str, Any]
    # The associated entity, e.g., if the event is IS_DRUG, then linked_entity is the Drug node
    linked_entity: Optional[Dict[str, Any]] = None

class HospitalAdmission(BaseModel):
    """Represents a hospital admission, which is the basic unit of a temporal trajectory"""
    node_id: int
    properties: Dict[str, Any]
    events: List[ClinicalEvent] = Field(default_factory=list)
    admission_time: str # Kept as string format for easier handling

class PatientTrajectory:
    """Represents a patient's complete treatment trajectory, composed of time-sorted hospital admissions"""
    def __init__(self, patient_id: str, admissions: List[HospitalAdmission]):
        self.patient_id = patient_id
        # Sort admission records by admission time
        self.admissions = sorted(admissions, key=lambda adm: adm.admission_time)
        # Flatten all events into a single chronologically sorted list
        self.flattened_events: List[ClinicalEvent] = self._flatten_events()

    def _flatten_events(self) -> List[ClinicalEvent]:
        """Flattens the events from all hospital admissions into a chronological sequence"""
        all_events = []
        for admission in self.admissions:
            # Assume the time for events within the same admission is the same as the admission time
            # If an event has a more precise timestamp, it should be used here
            for event in admission.events:
                event.properties['admission_time'] = admission.admission_time
                all_events.append(event)
        return all_events
    
    def __repr__(self):
        return f"PatientTrajectory(patient_id={self.patient_id}, num_admissions={len(self.admissions)}, num_events={len(self.flattened_events)})"

# --- Data Model for Alignment Results ---
class AlignmentResult(BaseModel):
    """Stores a pair of aligned entities and their related information"""
    tkg_event: ClinicalEvent
    gg_node: GuidelineNode
    initial_score: float  
    llm_score: Optional[float] = None # Score after LLM re-ranking
    llm_rationale: Optional[str] = None # Rationale given by the LLM