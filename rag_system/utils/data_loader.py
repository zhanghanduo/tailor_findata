import json
import os
import re
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_financial_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load financial data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing financial data.
        
    Returns:
        List of dictionaries containing the financial data.
    """
    logger.info(f"Loading financial data from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} documents")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def extract_tables_from_text(text: str, min_pipe_chars: int = 2) -> List[Dict[str, Any]]:
    """
    Extract tables from financial text.
    
    Args:
        text: The text containing potential tables.
        min_pipe_chars: Minimum number of pipe characters to consider a line as part of a table.
        
    Returns:
        List of dictionaries, each containing a table and its metadata.
    """
    lines = text.split('\n')
    tables = []
    current_table_lines = []
    in_table = False
    
    for line in lines:
        pipe_count = line.count('|')
        
        if pipe_count >= min_pipe_chars:
            if not in_table:
                in_table = True
                current_table_lines = []
            current_table_lines.append(line)
        else:
            if in_table and current_table_lines:
                table_text = '\n'.join(current_table_lines)
                try:
                    table_data = parse_pipe_table(table_text)
                    tables.append({
                        "type": "table",
                        "content": table_text,
                        "parsed_table": table_data
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse table: {str(e)}")
                    tables.append({
                        "type": "table",
                        "content": table_text,
                        "parsed_table": None
                    })
                in_table = False
                current_table_lines = []
    
    # Handle case where document ends with a table
    if in_table and current_table_lines:
        table_text = '\n'.join(current_table_lines)
        try:
            table_data = parse_pipe_table(table_text)
            tables.append({
                "type": "table",
                "content": table_text,
                "parsed_table": table_data
            })
        except Exception as e:
            logger.warning(f"Failed to parse table: {str(e)}")
            tables.append({
                "type": "table",
                "content": table_text,
                "parsed_table": None
            })
    
    return tables

def parse_pipe_table(table_text: str) -> pd.DataFrame:
    """
    Parse a pipe-delimited table into a pandas DataFrame.
    
    Args:
        table_text: Text of the table with pipe delimiters.
        
    Returns:
        pandas DataFrame representation of the table.
    """
    lines = [line.strip() for line in table_text.split('\n') if line.strip()]
    
    # Identify header separator line if it exists
    header_separator_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^[\-\|]+$', line.replace(' ', '')):
            header_separator_idx = i
            break
    
    # Extract rows
    rows = []
    for i, line in enumerate(lines):
        if i == header_separator_idx:
            continue
        # Split by pipe and strip whitespace
        cells = [cell.strip() for cell in line.split('|')]
        # Remove empty cells at start and end (from leading/trailing pipes)
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        if cells:
            rows.append(cells)
    
    # Create DataFrame
    if not rows:
        return pd.DataFrame()
    
    # Determine if the first row is a header
    is_header = header_separator_idx is not None and header_separator_idx == 1
    
    if is_header and len(rows) > 1:
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        df = pd.DataFrame(rows)
    
    return df

def split_document_with_tables(doc: Dict[str, Any], min_pipe_chars: int = 2) -> List[Dict[str, Any]]:
    """
    Split a document into text and table chunks.
    
    Args:
        doc: Document dictionary with 'content' field.
        min_pipe_chars: Minimum number of pipe characters to consider a line as part of a table.
        
    Returns:
        List of dictionaries, each containing either text or table content.
    """
    content = doc.get('content', '')
    metadata = doc.get('metadata', {})
    doc_id = doc.get('id', '')
    
    # Extract tables
    tables = extract_tables_from_text(content, min_pipe_chars)
    
    if not tables:
        # No tables found, return the original document
        return [{
            "id": doc_id,
            "content": content,
            "metadata": {**metadata, "type": "text"}
        }]
    
    # Split the content based on table locations
    text_chunks = []
    last_end = 0
    
    for table in tables:
        table_text = table["content"]
        table_start = content.find(table_text, last_end)
        
        if table_start > last_end:
            # Add text chunk before the table
            text_chunk = content[last_end:table_start].strip()
            if text_chunk:
                text_chunks.append({
                    "id": f"{doc_id}_text_{len(text_chunks)}",
                    "content": text_chunk,
                    "metadata": {**metadata, "type": "text"}
                })
        
        # Add the table chunk
        table_end = table_start + len(table_text)
        parsed_table = table.get("parsed_table")
        
        table_metadata = {
            **metadata, 
            "type": "table",
        }
        
        if parsed_table is not None:
            # Convert DataFrame to a list of dictionaries for JSON serialization
            table_metadata["columns"] = parsed_table.columns.tolist()
            table_metadata["row_count"] = len(parsed_table)
        
        text_chunks.append({
            "id": f"{doc_id}_table_{len(text_chunks)}",
            "content": table_text,
            "metadata": table_metadata
        })
        
        last_end = table_end
    
    # Add any remaining text after the last table
    if last_end < len(content):
        text_chunk = content[last_end:].strip()
        if text_chunk:
            text_chunks.append({
                "id": f"{doc_id}_text_{len(text_chunks)}",
                "content": text_chunk,
                "metadata": {**metadata, "type": "text"}
            })
    
    return text_chunks

def preprocess_data(data: List[Dict[str, Any]], min_pipe_chars: int = 2) -> List[Dict[str, Any]]:
    """
    Preprocess financial data by splitting documents into text and table chunks.
    
    Args:
        data: List of document dictionaries.
        min_pipe_chars: Minimum number of pipe characters to consider a line as part of a table.
        
    Returns:
        List of processed documents with separated text and table chunks.
    """
    processed_docs = []
    
    for doc in data:
        doc_chunks = split_document_with_tables(doc, min_pipe_chars)
        processed_docs.extend(doc_chunks)
    
    logger.info(f"Preprocessed {len(data)} documents into {len(processed_docs)} chunks")
    return processed_docs 