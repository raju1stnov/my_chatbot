from autogen import AssistantAgent, UserProxyAgent
from py2neo import Graph
import chromadb

class KnowledgeOrchestrator:
    def __init__(self):
        self.graph = Graph("bolt://neo4j:7687", auth=("neo4j", "password"))
        self.chroma = chromadb.PersistentClient(path="/app/chroma_db")
        self.llm = LLMWrapper("HuggingFaceH4/zephyr-7b-beta")
        
        self.query_analyzer = AssistantAgent(
            name="QueryAnalyzer",
            system_message="Analyze user questions to extract technical entities"
        )
        
        self.retriever = AssistantAgent(
            name="KnowledgeRetriever",
            system_message="Retrieve relevant context from knowledge graph and vector DB"
        )

    def process_query(self, question):
        # Step 1: Query Analysis
        entities = self.query_analyzer.analyze(question)
        
        # Step 2: Knowledge Retrieval
        context = self.retrieve_context(entities)
        
        # Step 3: Generate Response
        response = self.llm.generate(
            f"Answer this question: {question}\nUsing this context: {context}"
        )
        
        return {"answer": response, "sources": context['sources"]}

    def retrieve_context(self, entities):
        # Knowledge Graph Expansion
        query = f"""
        MATCH (e)-[r]-(related)
        WHERE e.name IN {entities}
        RETURN collect(distinct related.name) as context
        """
        kg_context = self.graph.run(query).data()
        
        # Vector DB Search
        collection = self.chroma.get_collection("main")
        results = collection.query(
            query_texts=[" ".join(entities + kg_context["context"])],
            n_results=5
        )
        
        return {
            "kg": kg_context,
            "vector": results,
            "sources": list(set([d['source'] for d in results['metadatas']]))
        }