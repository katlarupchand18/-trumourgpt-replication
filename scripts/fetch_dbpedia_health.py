#!/usr/bin/env python
"""
Fetch health-related triples from DBpedia using SPARQL.
Replaces hardcoded triples with authentic DBpedia data.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

try:
    from SPARQLWrapper import SPARQLWrapper, JSON
except ImportError:
    logger.error("SPARQLWrapper not installed. Run: pip install SPARQLWrapper")
    raise

# DBpedia SPARQL endpoint
DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"

# Health-related SPARQL queries
HEALTH_QUERIES = {
    "diseases": """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?disease ?label ?abstract ?cause ?treatment ?prevention WHERE {
            ?disease a dbo:Disease .
            ?disease rdfs:label ?label .
            FILTER(LANG(?label) = 'en')
            OPTIONAL { ?disease dbo:abstract ?abstract . FILTER(LANG(?abstract) = 'en') }
            OPTIONAL { ?disease dbo:cause ?cause . }
            OPTIONAL { ?disease dbo:treatment ?treatment . }
            OPTIONAL { ?disease dbo:prevention ?prevention . }
        }
        LIMIT 200
    """,

    "viruses": """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?virus ?label ?abstract ?transmission WHERE {
            ?virus a dbo:Virus .
            ?virus rdfs:label ?label .
            FILTER(LANG(?label) = 'en')
            OPTIONAL { ?virus dbo:abstract ?abstract . FILTER(LANG(?abstract) = 'en') }
            OPTIONAL { ?virus dbo:transmission ?transmission . }
        }
        LIMIT 100
    """,

    "vaccines": """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?vaccine ?label ?abstract ?targetDisease ?type WHERE {
            ?vaccine a dbo:Vaccine .
            ?vaccine rdfs:label ?label .
            FILTER(LANG(?label) = 'en')
            OPTIONAL { ?vaccine dbo:abstract ?abstract . FILTER(LANG(?abstract) = 'en') }
            OPTIONAL { ?vaccine dbo:targetDisease ?targetDisease . }
            OPTIONAL { ?vaccine dbo:vaccineType ?type . }
        }
        LIMIT 100
    """,

    "drugs": """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?drug ?label ?abstract ?treats ?sideEffect WHERE {
            ?drug a dbo:Drug .
            ?drug rdfs:label ?label .
            FILTER(LANG(?label) = 'en')
            OPTIONAL { ?drug dbo:abstract ?abstract . FILTER(LANG(?abstract) = 'en') }
            OPTIONAL { ?drug dbo:treats ?treats . }
            OPTIONAL { ?drug dbo:sideEffect ?sideEffect . }
        }
        LIMIT 150
    """,

    "health_organizations": """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?org ?label ?abstract ?location WHERE {
            ?org a dbo:Organisation .
            ?org rdfs:label ?label .
            FILTER(LANG(?label) = 'en')
            FILTER(REGEX(?label, "(WHO|CDC|FDA|NIH|EMA|Health|Medical|Hospital)", "i"))
            OPTIONAL { ?org dbo:abstract ?abstract . FILTER(LANG(?abstract) = 'en') }
            OPTIONAL { ?org dbo:location ?location . }
        }
        LIMIT 50
    """,

    "medical_treatments": """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?treatment ?label ?abstract ?treatsDisease WHERE {
            ?treatment a dbo:MedicalTreatment .
            ?treatment rdfs:label ?label .
            FILTER(LANG(?label) = 'en')
            OPTIONAL { ?treatment dbo:abstract ?abstract . FILTER(LANG(?abstract) = 'en') }
            OPTIONAL { ?treatment dbo:treatsDisease ?treatsDisease . }
        }
        LIMIT 100
    """,

    "symptoms": """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?symptom ?label ?abstract ?disease WHERE {
            ?symptom a dbo:Symptom .
            ?symptom rdfs:label ?label .
            FILTER(LANG(?label) = 'en')
            OPTIONAL { ?symptom dbo:abstract ?abstract . FILTER(LANG(?abstract) = 'en') }
            OPTIONAL { ?disease dbo:symptom ?symptom . }
        }
        LIMIT 100
    """,
}

def fetch_sparql(query: str, endpoint: str = DBPEDIA_ENDPOINT) -> List[Dict[str, Any]]:
    """Execute SPARQL query and return results."""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(120)

    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        logger.error(f"SPARQL query failed: {e}")
        return []

def extract_triples_from_results(results: List[Dict], category: str) -> List[Dict]:
    """Convert SPARQL results to triples."""
    triples = []

    for row in results:
        subject = row.get("disease") or row.get("virus") or row.get("vaccine") or \
                  row.get("drug") or row.get("org") or row.get("treatment") or row.get("symptom")
        label = row.get("label", {}).get("value", "")

        if not subject or not label:
            continue

        # Clean up URI to readable form
        subj_uri = subject.get("value", "")
        subj_name = subj_uri.split("/")[-1].replace("_", " ")

        # Extract relations from various properties
        for prop, prop_label in [
            ("cause", "causes"),
            ("treatment", "treats"),
            ("prevention", "prevents"),
            ("transmission", "transmitted via"),
            ("targetDisease", "targets"),
            ("vaccineType", "is type"),
            ("treats", "treats"),
            ("sideEffect", "has side effect"),
            ("location", "located in"),
            ("treatsDisease", "treats"),
            ("disease", "symptom of"),
        ]:
            if prop in row and row[prop]:
                obj_uri = row[prop].get("value", "")
                obj_name = obj_uri.split("/")[-1].replace("_", " ")
                triples.append({
                    "head": subj_name,
                    "relation": prop_label,
                    "tail": obj_name,
                    "source": f"DBpedia {category}",
                    "confidence": 0.85,
                    "head_uri": subj_uri,
                    "tail_uri": obj_uri,
                })

        # Add type triple
        triples.append({
            "head": subj_name,
            "relation": "is a",
            "tail": category.replace("_", " ").title(),
            "source": f"DBpedia {category}",
            "confidence": 0.9,
            "head_uri": subj_uri,
            "tail_uri": "",
        })

    return triples

def fetch_all_health_triples() -> Dict[str, List[Dict]]:
    """Fetch all health triples from DBpedia."""
    all_triples = {}

    for category, query in HEALTH_QUERIES.items():
        logger.info(f"Fetching {category} from DBpedia...")
        results = fetch_sparql(query)
        logger.info(f"  Got {len(results)} results for {category}")

        triples = extract_triples_from_results(results, category)
        logger.info(f"  Extracted {len(triples)} triples for {category}")

        all_triples[category] = triples
        time.sleep(1)  # Be nice to the endpoint

    return all_triples

def save_triples(triples: Dict[str, List[Dict]], output_path: str):
    """Save triples to JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(triples, f, indent=2)

    total = sum(len(t) for t in triples.values())
    logger.info(f"Saved {total} triples across {len(triples)} categories to {output}")

def main():
    logger.info("Fetching health triples from DBpedia...")

    triples = fetch_all_health_triples()

    output_path = "data/kb/dbpedia_health_triples.json"
    save_triples(triples, output_path)

    # Print summary
    for category, t in triples.items():
        print(f"  {category}: {len(t)} triples")

    total = sum(len(t) for t in triples.values())
    print(f"\nTotal: {total} triples")

if __name__ == "__main__":
    main()