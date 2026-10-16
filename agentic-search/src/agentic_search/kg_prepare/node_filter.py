"""
Node Filtering Module for Knowledge Graph Path Generation

Implements filtering criteria for selecting high-quality seed nodes:
- Out-degree constraints
- Relation entropy thresholds
- Label quality checks (uniqueness, non-generic)
"""

import re
from typing import Optional, List, Set
from dataclasses import dataclass


@dataclass
class NodeFilterConfig:
    """Configuration for node filtering criteria."""
    min_out_degree: int = 5
    max_out_degree: int = 50
    min_entropy: float = 1.5
    min_label_length: int = 3
    max_label_length: int = 100
    
    # Common words to filter out (too generic)
    stopwords: Set[str] = None
    
    # Patterns to exclude
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        if self.stopwords is None:
            self.stopwords = {
                "list", "index", "category", "disambiguation", "template",
                "file", "image", "wikipedia", "portal", "help", "talk",
                "user", "mediawiki", "module", "draft", "special"
            }
        
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                r"^\d+$",  # Pure numbers
                r"^[A-Z]$",  # Single letters
                r"^List of",  # List articles
                r"\(disambiguation\)",  # Disambiguation pages
                r"^Category:",  # Category pages
                r"^Template:",  # Template pages
                r"^File:",  # File pages
                r"^Wikipedia:",  # Wikipedia meta pages
                r"^Portal:",  # Portal pages
                r"^\d{4}$",  # Years
                r"^\d{4} in",  # Year-based articles
            ]


class NodeFilter:
    """
    Filter nodes based on quality criteria for path generation.
    """
    
    def __init__(self, config: Optional[NodeFilterConfig] = None):
        self.config = config or NodeFilterConfig()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.exclude_patterns
        ]
    
    def check_label_quality(self, title: str) -> bool:
        """
        Check if a label/title meets quality criteria.
        
        Returns True if the label is:
        - Not too short or too long
        - Not a common/generic word
        - Not matching excluded patterns (pure numbers, list pages, etc.)
        """
        if not title:
            return False
        
        title_lower = title.lower().strip()
        
        # Check length
        if len(title) < self.config.min_label_length:
            return False
        if len(title) > self.config.max_label_length:
            return False
        
        # Check for stopwords
        first_word = title_lower.split()[0] if title_lower.split() else ""
        if first_word in self.config.stopwords:
            return False
        
        # Check exclude patterns
        for pattern in self._compiled_patterns:
            if pattern.search(title):
                return False
        
        return True
    
    def check_degree_constraint(self, out_degree: int) -> bool:
        """Check if out-degree is within acceptable range."""
        return self.config.min_out_degree <= out_degree <= self.config.max_out_degree
    
    def check_entropy_threshold(self, entropy: float) -> bool:
        """Check if relation entropy meets threshold."""
        return entropy >= self.config.min_entropy
    
    def is_valid_seed_node(
        self,
        title: str,
        out_degree: int,
        entropy: float
    ) -> bool:
        """
        Check if a node is valid as a seed for path generation.
        
        Combines all quality checks:
        - Label quality
        - Degree constraints
        - Entropy threshold
        """
        if not self.check_label_quality(title):
            return False
        if not self.check_degree_constraint(out_degree):
            return False
        if not self.check_entropy_threshold(entropy):
            return False
        return True


# Stop relations that should be avoided in paths
RELATION_STOPLIST = {
    # Navigation and meta relations
    "see also",
    "external links",
    "references",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "footnotes",
    
    # Generic relations
    "main article",
    "this article",
    "citation needed",
    
    # Administrative
    "edit",
    "talk",
    "history",
    "wikipedia",
    
    # Common but uninformative
    "united states",
    "united kingdom",
    "new york",
    "london",
    "english",
    "american",
}


# Strict mode: relation filtering for higher-quality paths
STRICT_RELATION_BLOCKLIST = {
    "instance of",
    "subclass of",
    "main category",
    "category's main topic",
    "topic's main category",
    "different from",
}

STRICT_RELATION_BLOCKLIST_IDS = {
    "P31",   # instance of
    "P279",  # subclass of
    "P910",  # main category
    "P301",  # category's main topic
    "P1889", # different from
}

STRICT_RELATION_ALLOWLIST = {
    "author",
    "illustrator",
    "award received",
    "country of citizenship",
    "country of origin",
    "language of work or name",
    "language of work",
    "languages spoken",
    "place of birth",
    "place of death",
    "educated at",
    "occupation",
    "employer",
    "member of",
    "field of work",
    "director",
    "producer",
    "composer",
    "cast member",
    "publisher",
    "publication date",
    "inception",
    "headquarters location",
    "located in",
    "capital",
    "country",
    "continent",
    "has part",
    "part of",
    "named after",
    "creator",
    "developer",
}

STRICT_RELATION_ALLOWLIST_IDS = {
    "P50",   # author
    "P110",  # illustrator
    "P166",  # award received
    "P27",   # country of citizenship
    "P495",  # country of origin
    "P407",  # language of work or name
    "P1412", # languages spoken
    "P19",   # place of birth
    "P20",   # place of death
    "P69",   # educated at
    "P106",  # occupation
    "P108",  # employer
    "P463",  # member of
    "P101",  # field of work
    "P57",   # director
    "P162",  # producer
    "P86",   # composer
    "P161",  # cast member
    "P123",  # publisher
    "P577",  # publication date
    "P571",  # inception
    "P159",  # headquarters location
    "P131",  # located in the administrative territorial entity
    "P36",   # capital
    "P17",   # country
    "P30",   # continent
    "P527",  # has part
    "P361",  # part of
    "P138",  # named after
    "P170",  # creator
    "P178",  # developer
}

TYPE_LIFT_TERMS = {
    "class of",
    "type of",
    "category of",
    "taxonomy",
    "taxon",
    "wikimedia category",
}


def is_category_node(title: str, text: str = "") -> bool:
    """Heuristic check for Wikimedia category nodes."""
    title_lower = (title or "").lower().strip()
    if title_lower.startswith("category:"):
        return True
    text_lower = (text or "").lower()
    return "wikimedia category" in text_lower


def is_type_lift_node(title: str, text: str = "") -> bool:
    """Heuristic check for abstract/type-lift nodes."""
    combined = f"{title} {text}".lower()
    return any(term in combined for term in TYPE_LIFT_TERMS)


def is_relation_valid(
    anchor_text: str,
    target_title: str,
    target_text: str = "",
    property_id: str = "",
    strict: bool = False,
    allowlist: Optional[Set[str]] = None,
    blocklist: Optional[Set[str]] = None,
    exclude_category_nodes: bool = False,
    exclude_type_lift_nodes: bool = False
) -> bool:
    """
    Check if a relation (anchor) is valid for path traversal.
    
    Filters out:
    - Relations in stoplist
    - Generic/uninformative relations
    - Self-references
    """
    if not anchor_text or not target_title:
        return False
    
    anchor_lower = anchor_text.lower().strip()
    target_lower = target_title.lower().strip()
    
    # Check stoplist
    if anchor_lower in RELATION_STOPLIST:
        return False

    # Check if target is a list/category/etc
    for stopword in ["list of", "category:", "template:", "file:", "wikipedia:"]:
        if target_lower.startswith(stopword):
            return False

    if strict and exclude_category_nodes:
        if is_category_node(target_title, target_text):
            return False

    if strict and exclude_type_lift_nodes:
        if is_type_lift_node(target_title, target_text):
            return False

    if strict:
        property_id = (property_id or "").upper()
        block = blocklist if blocklist is not None else STRICT_RELATION_BLOCKLIST
        if anchor_lower in block or (property_id and property_id in STRICT_RELATION_BLOCKLIST_IDS):
            return False

        # Apply allowlist only when we have a property_id signal (Wikidata)
        use_allowlist = property_id.startswith("P")
        if use_allowlist:
            allow = allowlist if allowlist is not None else STRICT_RELATION_ALLOWLIST
            if allow:
                if anchor_lower not in allow and property_id not in STRICT_RELATION_ALLOWLIST_IDS:
                    return False
    
    # Check if it's a year or pure number
    if re.match(r"^\d+$", target_lower):
        return False
    
    return True


# Answer type classification
ANSWER_TYPE_PATTERNS = {
    "person": [
        r"\b(born|died|age|birthday)\b",
        r"\b(actor|actress|director|writer|author|scientist|politician)\b",
        r"\b(president|king|queen|prince|princess|emperor)\b",
    ],
    "organization": [
        r"\b(company|corporation|inc\.|ltd\.|llc)\b",
        r"\b(university|college|school|institute)\b",
        r"\b(organization|association|foundation|agency)\b",
    ],
    "location": [
        r"\b(city|town|village|country|state|province)\b",
        r"\b(river|mountain|lake|ocean|sea)\b",
        r"\b(located|situated|based in)\b",
    ],
    "date": [
        r"\b\d{4}\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b(century|decade|year|month|day)\b",
    ],
    "number": [
        r"\b\d+(\.\d+)?\s*(million|billion|thousand)\b",
        r"\b\d+(\.\d+)?\s*(km|m|cm|kg|lb|ft|mi)\b",
        r"\b(number|count|total|amount)\b",
    ],
}


def classify_answer_type(text: str, title: str) -> str:
    """
    Classify the type of an entity based on its text and title.
    
    Returns one of: person, organization, location, date, number, other
    """
    if not text and not title:
        return "other"
    
    combined = f"{title} {text}".lower()
    
    scores = {}
    for answer_type, patterns in ANSWER_TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                score += 1
        scores[answer_type] = score
    
    if not scores or max(scores.values()) == 0:
        return "other"
    
    return max(scores, key=scores.get)


def is_same_answer_type(type1: str, type2: str) -> bool:
    """Check if two answer types are compatible for distractor generation."""
    if type1 == type2:
        return True
    
    # Allow some flexibility
    compatible_groups = [
        {"person", "other"},
        {"organization", "other"},
        {"location", "other"},
        {"date", "number"},
    ]
    
    for group in compatible_groups:
        if type1 in group and type2 in group:
            return True
    
    return False

