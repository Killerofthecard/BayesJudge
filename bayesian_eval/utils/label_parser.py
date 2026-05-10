"""
LLM response label parser.
"""


class LabelParser:
    """Label parser."""

    VALID_LABELS = {'A', 'B', 'T'}

    @classmethod
    def parse(cls, response: str) -> str:
        """
        Parse an LLM response into A / B / T.

        Args:
            response: Raw LLM response.

        Returns:
            'A', 'B', or 'T' (for ties / unparseable responses).
        """
        if not response:
            return 'T'

        resp_clean = response.strip().upper()

        if resp_clean in ['A', 'B']:
            return resp_clean

        if resp_clean in ['T', 'TIE', 'C']:
            return 'T'

        if resp_clean.startswith('A'):
            return 'A'
        if resp_clean.startswith('B'):
            return 'B'

        return 'T'

    @classmethod
    def parse_batch(cls, responses: list) -> list:
        """Parse a batch of responses."""
        return [cls.parse(r) for r in responses]


def parse_llm_label(response: str) -> str:
    """Parse an LLM response into A / B / T."""
    return LabelParser.parse(response)
