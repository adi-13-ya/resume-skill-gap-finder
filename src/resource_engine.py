"""
resource_engine.py — Phase 4: Resource Recommendation
=====================================================
Builds a "Resource Mapping" function. For every missing skill identified,
the system queries a pre-defined list of links or generates a dynamic
search query to fetch relevant study materials.

Public API:
    get_resources_for_skill(skill_name)
        → dict containing links to study materials
"""

import urllib.parse
from typing import Any

# A curated dictionary of high-quality resources for common skills
# In a real-world scenario, this might be powered by the Coursera/Udemy/YouTube API.
_CURATED_RESOURCES = {
    "python": [
        {"platform": "Coursera", "title": "Python for Everybody Specialization", "url": "https://www.coursera.org/specializations/python"},
        {"platform": "YouTube", "title": "Python Tutorial for Beginners - Full Course", "url": "https://www.youtube.com/watch?v=_uQrJ0TkZlc"}
    ],
    "java": [
        {"platform": "Coursera", "title": "Object Oriented Programming in Java", "url": "https://www.coursera.org/specializations/object-oriented-programming"},
        {"platform": "YouTube", "title": "Java Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=eIrMbAQSU34"}
    ],
    "sql": [
        {"platform": "Coursera", "title": "Learn SQL Basics for Data Science", "url": "https://www.coursera.org/specializations/learn-sql-basics-data-science"},
        {"platform": "YouTube", "title": "SQL Tutorial - Full Database Course for Beginners", "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY"}
    ],
    "aws": [
        {"platform": "Coursera", "title": "AWS Fundamentals Specialization", "url": "https://www.coursera.org/specializations/aws-fundamentals"},
        {"platform": "AWS Skill Builder", "title": "AWS Cloud Practitioner Essentials", "url": "https://explore.skillbuilder.aws/learn/course/external/view/elearning/134/aws-cloud-practitioner-essentials"}
    ],
    "docker": [
        {"platform": "Udemy", "title": "Docker Mastery", "url": "https://www.udemy.com/course/docker-mastery/"},
        {"platform": "YouTube", "title": "Docker Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=pTFZFxd4hOI"}
    ],
    "kubernetes": [
        {"platform": "Coursera", "title": "Architecting with Google Kubernetes Engine", "url": "https://www.coursera.org/specializations/architecting-google-kubernetes-engine"},
        {"platform": "YouTube", "title": "Kubernetes Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=X48VuDVv0do"}
    ],
    "machine learning": [
        {"platform": "Coursera", "title": "Machine Learning Specialization by Andrew Ng", "url": "https://www.coursera.org/specializations/machine-learning-introduction"},
        {"platform": "YouTube", "title": "Machine Learning Course for Beginners", "url": "https://www.youtube.com/watch?v=NWONeJKn6kc"}
    ],
    "react": [
        {"platform": "Coursera", "title": "Front-End Web Development with React", "url": "https://www.coursera.org/learn/front-end-react"},
        {"platform": "YouTube", "title": "React Full Course", "url": "https://www.youtube.com/watch?v=bMknfKXIFA8"}
    ]
}

def get_resources_for_skill(skill_name: str) -> dict[str, Any]:
    """Fetch study materials for a given skill.
    
    If the skill exists in the curated list, it returns those links.
    Otherwise, it generates dynamic search URLs for YouTube and Coursera.
    
    Args:
        skill_name: The name of the missing skill.
        
    Returns:
        A dictionary containing the skill name and a list of resources.
    """
    normalized_skill = skill_name.lower().strip()
    
    # Check if we have curated resources
    for key, resources in _CURATED_RESOURCES.items():
        # Match if the key is in the skill name (e.g., 'amazon web services (aws)' matches 'aws')
        # Or if the skill name is in the key
        if key in normalized_skill or normalized_skill in key:
            return {
                "skill": skill_name,
                "curated": True,
                "resources": resources
            }
            
    # Fallback to generating dynamic search links
    query = urllib.parse.quote(f"{skill_name} tutorial full course")
    coursera_query = urllib.parse.quote(skill_name)
    
    fallback_resources = [
        {
            "platform": "YouTube",
            "title": f"Search: {skill_name} tutorials",
            "url": f"https://www.youtube.com/results?search_query={query}"
        },
        {
            "platform": "Coursera",
            "title": f"Search: {skill_name} courses",
            "url": f"https://www.coursera.org/search?query={coursera_query}"
        }
    ]
    
    return {
        "skill": skill_name,
        "curated": False,
        "resources": fallback_resources
    }
