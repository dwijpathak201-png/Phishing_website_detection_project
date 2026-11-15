"FIRST OF ALL, HAM LOG URL KA STRUCTURE CHECK KAREGA KI USKA BASIC COMPONENTS PRESENT HA KI NHI - JASA DOMAIN NAME"
from urllib.parse import urlparse

def validate_url(url):
    """
    Validates whether the user input looks like a website URL.

    Returns:
        (True, normalized_url)  -> if valid
        (False, error_message)  -> if invalid
    """

    # Remove unnecessary spaces
    url = url.strip()

    # 1. Check if input is empty
    if not url:
        return False, "Please enter a URL."

    # 2. Add HTTPS if the user didn't provide a protocol
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # 3. Parse the URL
    parsed = urlparse(url)

    # 4. Check whether a domain exists
    if not parsed.netloc:
        return False, "Please enter a valid website URL."

    # 5. Basic domain check
    if "." not in parsed.netloc:
        return False, "Invalid domain name."

    # URL passed validation
    return True, url