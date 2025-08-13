from langchain.tools import tool
import requests
from xml.etree import ElementTree

@tool
def pubmed_lookup(keyword: str) -> str:
    """
    Searches PubMed for articles matching the keyword and returns the abstract of the top result.
    Input: keyword (e.g., "Alzheimer's disease")
    Output: abstract text of the top article
    """
    # Step 1: Search for article IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": keyword,
        "retmax": 1,
        "retmode": "json"
    }
    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()
    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    
    if not id_list:
        return "No articles found for the given keyword."

    # Step 2: Fetch article details
    article_id = id_list[0]
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": article_id,
        "retmode": "xml"
    }
    fetch_response = requests.get(fetch_url, params=fetch_params)
    tree = ElementTree.fromstring(fetch_response.content)

    # Extract abstract text
    abstract_texts = tree.findall(".//AbstractText")
    abstract = " ".join([elem.text for elem in abstract_texts if elem.text])

    return abstract if abstract else "No abstract available for the top result."
