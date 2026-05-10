"""
Data type definitions.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


class LabelMapping:
    """Label mapping constants."""
    TO_INDEX = {'A': 0, 'C': 1, 'B': 2}
    FROM_INDEX = {0: 'A', 1: 'C', 2: 'B'}

    @classmethod
    def to_index(cls, label: str) -> int:
        """Map a label to its index."""
        return cls.TO_INDEX.get(label.upper(), 1)

    @classmethod
    def from_index(cls, index: int) -> str:
        """Map an index back to its label."""
        return cls.FROM_INDEX.get(index, 'T')


@dataclass
class SampleData:
    """
    Evaluation sample data structure.

    Attributes:
        human_judgments: Human annotations, format {rater_id: label}.
        llm_judgments: LLM swap annotations, format
            {llm_id: (label_AB, label_BA)}.
        metadata: Optional metadata (e.g. question_id, model_a, model_b).
        conversation_a: Conversation content for model A.
        conversation_b: Conversation content for model B.
    """
    human_judgments: Dict[str, str]
    llm_judgments: Dict[str, Tuple[str, str]]
    metadata: Optional[Dict] = None
    conversation_a: Optional[List[Dict[str, str]]] = None
    conversation_b: Optional[List[Dict[str, str]]] = None

    def get_question_id(self) -> str:
        """Return the question ID if available, otherwise 'N/A'."""
        if self.metadata and 'question_id' in self.metadata:
            return self.metadata['question_id']
        return 'N/A'

    def has_llm_judgment(self, llm_id: str) -> bool:
        """Check whether a judgment from the given LLM exists."""
        return llm_id in self.llm_judgments

    def get_llm_judgment(self, llm_id: str) -> Optional[Tuple[str, str]]:
        """Return the judgment from the given LLM, or None if absent."""
        return self.llm_judgments.get(llm_id)
