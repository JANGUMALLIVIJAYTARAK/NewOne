# kg_service.py
"""
Knowledge Graph Service: Handles entity/relation extraction and KG queries.
"""

import spacy
from neo4j import GraphDatabase

class KnowledgeGraphService:
    def __init__(self, neo4j_uri="bolt://localhost:7687", user="neo4j", password="password"):
        # Placeholder for Neo4j or other KG backend connection
        self.neo4j_uri = neo4j_uri
        self.user = user
        self.password = password
        # Load spaCy English model for entity extraction
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.user, self.password))

    def extract_entities_and_relations(self, text):
        """
        Extract entities and relationships from text using spaCy.
        """
        doc = self.nlp(text)
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]
        # Placeholder: No relation extraction in spaCy small model
        relations = []
        return {"entities": entities, "relations": relations}

    def add_to_kg(self, entities, relations):
        """
        Add extracted entities and relations to the KG backend.
        """
        with self.driver.session() as session:
            for ent in entities:
                session.run(
                    "MERGE (e:Entity {text: $text, label: $label})",
                    text=ent["text"], label=ent["label"]
                )
            # Relations can be added here if extracted
        return True

    def query_kg(self, query):
        """
        Query the KG for information. Returns empty results if Neo4j is unavailable.
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (e:Entity) WHERE toLower(e.text) CONTAINS toLower($query) RETURN e.text AS text, e.label AS label",
                    {"query": query}
                )
                results = [{"text": record["text"], "label": record["label"]} for record in result]
            return {"results": results}
        except Exception as e:
            # Log the error and return empty results
            import logging
            logging.getLogger(__name__).warning(f"KG query failed (Neo4j down?): {e}")
            return {"results": []}

# Example usage (to be removed in production):
if __name__ == "__main__":
    kg = KnowledgeGraphService()
    text = "Barack Obama was born in Hawaii. He was president of the USA."
    result = kg.extract_entities_and_relations(text)
    print(result)
