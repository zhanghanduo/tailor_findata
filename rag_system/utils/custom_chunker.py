from typing import List, Dict, Any, Optional, Callable
import re
import logging
from llama_index.schema import Document, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.node_parser import SentenceSplitter
import pandas as pd

logger = logging.getLogger(__name__)

def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract numerical values from text.
    
    Args:
        text: The text to extract numerical values from.
        
    Returns:
        List of numerical values extracted from the text.
    """
    # Pattern to match currency values and percentages
    # Handles formats like $1,234.56, 1,234.56, 1234.56, 12%, etc.
    pattern = r'(?:\$\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*%)?'
    
    matches = re.findall(pattern, text)
    numbers = []
    
    for match in matches:
        # Remove commas and convert to float
        try:
            clean_num = match.replace(',', '')
            numbers.append(float(clean_num))
        except ValueError:
            continue
    
    return numbers

class FinancialTextSplitter:
    """
    Custom text splitter for financial documents that preserves tables and 
    handles numerical data appropriately.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        table_detection_threshold: int = 2
    ):
        """
        Initialize the financial text splitter.
        
        Args:
            chunk_size: Maximum size of text chunks.
            chunk_overlap: Overlap between consecutive chunks.
            table_detection_threshold: Minimum number of pipe characters to consider a line as part of a table.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_detection_threshold = table_detection_threshold
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[TextNode]:
        """
        Split documents into nodes, preserving tables and extracting numerical data.
        
        Args:
            documents: List of document dictionaries.
            
        Returns:
            List of TextNode objects.
        """
        nodes = []
        
        for doc in documents:
            doc_nodes = self._split_document(doc)
            nodes.extend(doc_nodes)
        
        logger.info(f"Split {len(documents)} documents into {len(nodes)} nodes")
        return nodes
    
    def _split_document(self, doc: Dict[str, Any]) -> List[TextNode]:
        """
        Split a single document into nodes.
        
        Args:
            doc: Document dictionary.
            
        Returns:
            List of TextNode objects.
        """
        doc_type = doc.get("metadata", {}).get("type", "text")
        doc_id = doc.get("id", "")
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        
        if doc_type == "table":
            # For table documents, create a single node
            numbers = extract_numbers_from_text(content)
            table_metadata = {
                **metadata,
                "is_table": True,
                "contains_numbers": len(numbers) > 0,
                "number_count": len(numbers),
                "numbers": numbers if len(numbers) <= 10 else numbers[:10]  # Limit number of stored values
            }
            
            node = TextNode(
                text=content,
                metadata=table_metadata,
                node_id=doc_id
            )
            return [node]
        else:
            # For text documents, use sentence splitter
            llama_doc = Document(text=content, metadata=metadata, id_=doc_id)
            text_nodes = self.sentence_splitter.get_nodes_from_documents([llama_doc])
            
            # Enhance text nodes with numerical data
            for i, node in enumerate(text_nodes):
                numbers = extract_numbers_from_text(node.text)
                node.metadata.update({
                    "is_table": False,
                    "contains_numbers": len(numbers) > 0,
                    "number_count": len(numbers),
                    "numbers": numbers if len(numbers) <= 10 else numbers[:10]  # Limit number of stored values
                })
                
                # Generate a deterministic node ID
                if not node.id_:
                    node.id_ = f"{doc_id}_text_chunk_{i}"
            
            return text_nodes

def create_hierarchical_nodes(nodes: List[TextNode]) -> List[TextNode]:
    """
    Create hierarchical relationships between nodes from the same document.
    
    Args:
        nodes: List of TextNode objects.
        
    Returns:
        List of TextNode objects with hierarchical relationships.
    """
    # Group nodes by document ID (remove _text_chunk_X suffix)
    doc_id_pattern = r'(.+?)(?:_text_|_table_).+'
    doc_groups = {}
    
    for node in nodes:
        match = re.match(doc_id_pattern, node.id_)
        if match:
            doc_id = match.group(1)
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(node)
    
    # Add relationships between nodes in the same document
    for doc_id, doc_nodes in doc_groups.items():
        # Add parent-child relationships between nodes
        for i, node in enumerate(doc_nodes):
            for j, other_node in enumerate(doc_nodes):
                if i != j:
                    # Add bidirectional relationships
                    node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                        node_id=other_node.id_,
                        metadata={"doc_id": doc_id, "distance": abs(i - j)}
                    )
    
    return nodes 