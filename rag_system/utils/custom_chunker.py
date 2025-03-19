from typing import List, Dict, Any, Optional, Callable
import re
import logging
import json
from llama_index.core.schema import Document, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter
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

def extract_table_content_from_json(table_data: List[List[str]]) -> str:
    """
    Extract textual content from table data in JSON format.
    
    Args:
        table_data: Table data as a list of lists (rows and columns).
        
    Returns:
        Extracted text from the table.
    """
    if not table_data or not isinstance(table_data, list):
        return ""
    
    # Flatten the table and join all cells with spaces
    text_content = []
    for row in table_data:
        if isinstance(row, list):
            text_content.extend([str(cell) for cell in row])
    
    return " ".join(text_content)

class FinancialTextSplitter:
    """
    Custom text splitter for financial documents that handles various components
    (pre_text, table_markdown, table_data, post_text) differently.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        table_detection_threshold: int = 2,
        max_metadata_size: int = 256
    ):
        """
        Initialize the financial text splitter.
        
        Args:
            chunk_size: Maximum size of text chunks.
            chunk_overlap: Overlap between consecutive chunks.
            table_detection_threshold: Minimum number of pipe characters to consider a line as part of a table.
            max_metadata_size: Maximum size of metadata to store per node.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_detection_threshold = table_detection_threshold
        self.max_metadata_size = max_metadata_size
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # Dictionary to store full content and raw content mapped by node_id
        self.content_store = {}
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[TextNode]:
        """
        Split documents into nodes, handling different document types appropriately.
        
        Args:
            documents: List of document dictionaries.
            
        Returns:
            List of TextNode objects.
        """
        # Clear the content store before processing new documents
        self.content_store = {}
        nodes = []
        
        for doc in documents:
            doc_nodes = self._split_document(doc)
            nodes.extend(doc_nodes)
        
        logger.info(f"Split {len(documents)} documents into {len(nodes)} nodes")
        return nodes
    
    def _split_document(self, doc: Dict[str, Any]) -> List[TextNode]:
        """
        Split a single document into nodes based on its type.
        
        Args:
            doc: Document dictionary.
            
        Returns:
            List of TextNode objects.
        """
        doc_type = doc.get("metadata", {}).get("type", "text")
        doc_id = doc.get("id", "")
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        
        # Store full content and raw content in content_store
        full_content = metadata.pop("full_content", content)
        raw_content = metadata.pop("raw_content", None)
        raw_table = metadata.pop("raw_table", None)
        
        # Look for financial keywords in the content to enhance metadata
        financial_keywords = [
            "cash flow", "operating", "investing", "financing", "activities",
            "balance sheet", "income statement", "net income", "revenue", "expense",
            "capital", "dividend", "equity", "debt", "asset", "liability"
        ]
        
        found_keywords = []
        for keyword in financial_keywords:
            if re.search(r"\b" + keyword + r"\b", content, re.IGNORECASE):
                found_keywords.append(keyword)
        
        if doc_type in ["table_markdown", "table_data"]:
            # Extract numbers with their context to better understand the table
            numbers = extract_numbers_from_text(content)
            stored_numbers = numbers[:10] if numbers else []  # Store more numbers for tables
            
            # Extract patterns like "cash flow from operations: $12,345"
            financial_amounts = []
            amount_patterns = [
                r"(cash\s+(?:flow|provided|used|from|in)\s+\w+)\s*[:\-]?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
                r"((?:operating|investing|financing)\s+activities)\s*[:\-]?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
                r"((?:net|total|gross)\s+\w+)\s*[:\-]?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
            ]
            
            for pattern in amount_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 2:
                        desc, amount = match
                        try:
                            amount_num = float(amount.replace(',', '').replace('$', ''))
                            financial_amounts.append({
                                "description": desc.strip().lower(),
                                "amount": amount_num
                            })
                        except ValueError:
                            pass
            
            # Check if table contains year columns (common in financial statements)
            year_pattern = r"\b(19\d{2}|20\d{2})\b"
            years_mentioned = re.findall(year_pattern, content)
            
            # Extract financial metrics - key terms that often appear in cash flow statements
            financial_metrics = []
            metric_patterns = [
                r"net cash (?:from|provided by|used in) (?:operating|investing|financing) activities",
                r"cash (?:provided by|used in) (?:operating|investing|financing) activities",
                r"cash flows? from (?:operating|investing|financing) activities",
                r"net income",
                r"change in receivables",
                r"change in (?:inventory|inventories)",
                r"change in (?:accounts payable|payables)",
                r"change in (?:other assets|other liabilities)",
                r"depreciation and amortization",
                r"(?:capital|property|equipment) expenditures",
                r"dividends? paid",
                r"repurchase of common stock",
                r"proceeds from (?:issuance|sale)",
                r"payments? of debt",
                r"non-cash expenses"
            ]
            
            for pattern in metric_patterns:
                matches = re.findall(pattern, content.lower())
                financial_metrics.extend(matches)
            
            # Create enhanced metadata for table nodes
            table_metadata = {
                **metadata,
                "is_table": True,
                "table_type": doc_type,
                "contains_numbers": len(numbers) > 0,
                "number_count": len(numbers),
                "numbers_str": ",".join(map(str, stored_numbers)) if stored_numbers else "",
                "contains_years": bool(years_mentioned),
                "years_mentioned": ",".join(sorted(set(years_mentioned))) if years_mentioned else "",
                "financial_keywords": ",".join(found_keywords) if found_keywords else "",
                "financial_metrics": ",".join(set(financial_metrics)) if financial_metrics else "",
                "financial_amounts": json.dumps(financial_amounts[:5]) if financial_amounts else ""
            }
            
            if doc_type == "table_data" and raw_table:
                if isinstance(raw_table, list) and raw_table:
                    table_metadata["table_dimensions"] = f"{len(raw_table)}x{len(raw_table[0]) if raw_table[0] else 0}"
            
            # Create node with enhanced text representation
            # Add a plain text summary of the content to improve vector search
            text_content = content
            if financial_amounts:
                amounts_summary = "; ".join([f"{item['description']}: {item['amount']}" for item in financial_amounts[:5]])
                text_content = f"{text_content}\n\nTable contains financial amounts: {amounts_summary}"
            
            if years_mentioned:
                text_content = f"{text_content}\n\nTable includes data for years: {', '.join(sorted(set(years_mentioned)))}"
            
            node = TextNode(
                text=text_content,
                metadata=table_metadata,
                node_id=doc_id
            )
            
            # Store the content and original numbers in content_store
            self.content_store[doc_id] = {
                "full_content": full_content,
                "raw_content": raw_content,
                "raw_table": raw_table,
                "numbers": stored_numbers,
                "financial_amounts": financial_amounts if financial_amounts else None
            }
            
            return [node]
        
        elif doc_type in ["pre_text", "post_text"]:
            llama_doc = Document(text=content, metadata=metadata, id_=doc_id)
            text_nodes = self.sentence_splitter.get_nodes_from_documents([llama_doc])
            
            for i, node in enumerate(text_nodes):
                node_id = f"{doc_id}_chunk_{i}"
                numbers = extract_numbers_from_text(node.text)
                stored_numbers = numbers[:5] if numbers else []
                
                # Check if node contains financial keywords
                node_keywords = []
                for keyword in financial_keywords:
                    if re.search(r"\b" + keyword + r"\b", node.text, re.IGNORECASE):
                        node_keywords.append(keyword)
                
                node.metadata.update({
                    "is_table": False,
                    "text_type": doc_type,
                    "contains_numbers": len(numbers) > 0,
                    "number_count": len(numbers),
                    "numbers_str": ",".join(map(str, stored_numbers)) if stored_numbers else "",
                    "financial_keywords": ",".join(node_keywords) if node_keywords else ""
                })
                
                if not node.id_:
                    node.id_ = node_id
                
                # Store the content and original numbers in content_store
                self.content_store[node_id] = {
                    "full_content": full_content,
                    "raw_content": raw_content,
                    "numbers": stored_numbers
                }
            
            return text_nodes
        
        else:
            llama_doc = Document(text=content, metadata=metadata, id_=doc_id)
            text_nodes = self.sentence_splitter.get_nodes_from_documents([llama_doc])
            
            for i, node in enumerate(text_nodes):
                node_id = f"{doc_id}_chunk_{i}"
                numbers = extract_numbers_from_text(node.text)
                stored_numbers = numbers[:5] if numbers else []
                
                # Check if node contains financial keywords
                node_keywords = []
                for keyword in financial_keywords:
                    if re.search(r"\b" + keyword + r"\b", node.text, re.IGNORECASE):
                        node_keywords.append(keyword)
                
                node.metadata.update({
                    "is_table": False,
                    "contains_numbers": len(numbers) > 0,
                    "number_count": len(numbers),
                    "numbers_str": ",".join(map(str, stored_numbers)) if stored_numbers else "",
                    "financial_keywords": ",".join(node_keywords) if node_keywords else ""
                })
                
                if not node.id_:
                    node.id_ = node_id
                
                # Store the content and original numbers in content_store
                self.content_store[node_id] = {
                    "full_content": full_content,
                    "raw_content": raw_content,
                    "numbers": stored_numbers
                }
            
            return text_nodes
    
    def get_node_content(self, node_id: str) -> Dict[str, Any]:
        """
        Get the full content and raw content for a given node ID.
        
        Args:
            node_id: The ID of the node to get content for.
            
        Returns:
            Dictionary containing full_content and raw_content for the node.
        """
        return self.content_store.get(node_id, {})
    
    def get_node_context(self, node: TextNode) -> Dict[str, Any]:
        """
        Get the context information for a given node, including its content and metadata.
        
        Args:
            node: The TextNode to get context for.
            
        Returns:
            Dictionary containing node context information.
        """
        node_content = self.get_node_content(node.id_)
        return {
            "text": node.text,
            "metadata": node.metadata,
            **node_content
        }

    def get_node_numbers(self, node_id: str) -> List[float]:
        """
        Get the original numbers list for a given node ID.
        
        Args:
            node_id: The ID of the node to get numbers for.
            
        Returns:
            List of numbers extracted from the node's text.
        """
        content = self.content_store.get(node_id, {})
        return content.get("numbers", [])

def create_hierarchical_nodes(nodes: List[TextNode]) -> List[TextNode]:
    """
    Create hierarchical relationships between nodes from the same document.
    
    Args:
        nodes: List of TextNode objects.
        
    Returns:
        List of TextNode objects with hierarchical relationships.
    """
    # Group nodes by document ID (extract base document ID without component suffixes)
    doc_id_pattern = r'(.+?)(?:_pre_text|_table_markdown|_table_data|_post_text|_chunk_).*'
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
        # Sort nodes by type to maintain logical order: pre_text -> table -> post_text
        def get_node_order(node):
            if "pre_text" in node.id_:
                return 0
            elif "table" in node.id_:
                return 1
            elif "post_text" in node.id_:
                return 2
            else:
                return 3
        
        sorted_nodes = sorted(doc_nodes, key=get_node_order)
        
        # Add parent-child relationships between nodes
        for i, node in enumerate(sorted_nodes):
            for j, other_node in enumerate(sorted_nodes):
                if i != j:
                    # Add bidirectional relationships
                    node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                        node_id=other_node.id_,
                        metadata={"doc_id": doc_id, "distance": abs(i - j)}
                    )
    
    return nodes 