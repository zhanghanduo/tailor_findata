import json
import os
import re
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm

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
    table_start_idx = -1
    
    # More aggressive pattern to detect tables beyond just counting pipes
    # This also detects grid-like patterns that may be tables without pipe chars
    def is_potential_table_line(line):
        # Check for regular pipe-based tables
        if line.count('|') >= min_pipe_chars:
            return True
        
        # Check for aligned columns using spaces (common in financial reports)
        space_groups = re.findall(r'\s{2,}', line)
        if len(space_groups) >= 2 and any(len(g) >= 3 for g in space_groups):
            return True
            
        # Check for numeric alignments - lines with multiple numbers might be table rows
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', line)
        if len(numbers) >= 3:  # If multiple numbers appear in a line
            return True
        
        # Check for working capital terms with adjacent numbers
        wc_terms = ["accounts receivable", "inventory", "inventories", "accounts payable", 
                    "working capital", "current assets", "current liabilities"]
        
        for term in wc_terms:
            if term in line.lower():
                # If a working capital term is found and there's at least one number, treat as potential table
                if len(numbers) >= 1:
                    return True
            
        return False
    
    # First pass: identify potential table regions using the enhanced detection
    for i, line in enumerate(lines):
        is_table_line = is_potential_table_line(line)
        
        if is_table_line:
            if not in_table:
                in_table = True
                table_start_idx = i
                current_table_lines = []
            current_table_lines.append(line)
        else:
            # Only end a table if we've had at least 2 non-table lines or reached the end
            # This helps handle small gaps/formatting issues in tables
            if in_table:
                if i < len(lines) - 1 and not is_potential_table_line(lines[i+1]):
                    # Check if we have enough table lines to be meaningful
                    if len(current_table_lines) >= 2:
                        table_text = '\n'.join(current_table_lines)
                        try:
                            table_data = parse_pipe_table(table_text)
                            # Only save if it parsed into at least 2x2 cells
                            if not table_data.empty and len(table_data) > 1 and len(table_data.columns) > 1:
                                tables.append({
                                    "type": "table",
                                    "content": table_text,
                                    "parsed_table": table_data
                                })
                            else:
                                # Try parsing with more flexible approach for borderline cases
                                # If it's not a pipe table, maybe it's space-aligned
                                if '|' not in table_text and any(re.search(r'\s{3,}', l) for l in current_table_lines):
                                    # Log potential table that might need alternative parsing
                                    logger.debug(f"Detected potential space-aligned table: {table_text[:100]}...")
                                    tables.append({
                                        "type": "table",
                                        "content": table_text,
                                        "parsed_table": None
                                    })
                        except Exception as e:
                            logger.warning(f"Failed to parse table: {str(e)}")
                            # Still keep it as a table for retrieval
                            tables.append({
                                "type": "table",
                                "content": table_text,
                                "parsed_table": None
                            })
                    in_table = False
                    current_table_lines = []
    
    # Handle case where document ends with a table
    if in_table and current_table_lines:
        if len(current_table_lines) >= 2:  # Only process if we have at least 2 lines
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
    
    # Log statistics about extracted tables
    if tables:
        logger.info(f"Extracted {len(tables)} tables from text")
        parsed_count = sum(1 for t in tables if t["parsed_table"] is not None)
        logger.info(f"Successfully parsed {parsed_count}/{len(tables)} tables")
    else:
        logger.info("No tables extracted from text")
    
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
    Preprocess financial data by splitting documents into pre_text, table, and post_text chunks.
    Handles both table_markdown and table_data formats.
    
    Args:
        data: List of document dictionaries.
        min_pipe_chars: Minimum number of pipe characters to consider a line as part of a table.
        
    Returns:
        List of processed documents.
    """
    processed_docs = []
    
    # Add progress bar for document processing
    for doc in tqdm(data, desc="Processing documents"):
        doc_id = doc.get('id', '')
        metadata = doc.get('metadata', {})
        
        # Store the original full content in metadata for context
        original_content = doc.get('content', '')
        enhanced_metadata = {
            **metadata,
            "full_content": original_content,
            "doc_id": doc_id
        }
        
        # Process pre_text if available
        pre_text = doc.get('pre_text', '')
        if pre_text:
            pre_text_doc = {
                "id": f"{doc_id}_pre_text",
                "content": pre_text,
                "metadata": {
                    **enhanced_metadata,
                    "type": "pre_text"
                }
            }
            processed_docs.append(pre_text_doc)
        
        # Process table in both markdown and data formats
        table_markdown = doc.get('table_markdown', '')
        table_data = doc.get('table_data', [])
        
        if table_markdown:
            # Add explicit 'table' tag to type
            table_markdown_doc = {
                "id": f"{doc_id}_table_markdown",
                "content": table_markdown,
                "metadata": {
                    **enhanced_metadata,
                    "type": "table_markdown",
                    "is_table": True
                }
            }
            
            # Check for financial line items and add to metadata for easier retrieval
            financial_line_items = []
            
            # Look for key financial metrics
            if "net cash from operating activities" in table_markdown.lower():
                financial_line_items.append("net_cash_operating")
                table_markdown_doc["metadata"]["contains_net_cash_operating"] = True
                
                # Try to extract the specific values with years
                lines = table_markdown.lower().split('\n')
                for i, line in enumerate(lines):
                    if "net cash from operating activities" in line:
                        # Try to find values in this row
                        table_markdown_doc["metadata"]["net_cash_operating_line"] = i
                        
                        # Look for years and values in this row
                        years_values = {}
                        year_pattern = r"\b(19\d{2}|20\d{2})\b"
                        years_in_line = re.findall(year_pattern, line)
                        if years_in_line:
                            for year in years_in_line:
                                years_values[year] = "present_in_line"
                        
                        # If no years in line, check header rows for years
                        else:
                            for j in range(max(0, i-3), i):
                                if j < len(lines):
                                    years_in_header = re.findall(year_pattern, lines[j])
                                    if years_in_header:
                                        # Found potential header with years
                                        # Try to match values based on position
                                        headers = re.split(r'\s{2,}|\t|,|\|', lines[j])
                                        values = re.split(r'\s{2,}|\t|,|\|', line)
                                        
                                        for k, header_item in enumerate(headers):
                                            years_in_item = re.findall(year_pattern, header_item)
                                            if years_in_item and k < len(values):
                                                years_values[years_in_item[0]] = values[k].strip()
                        
                        if years_values:
                            table_markdown_doc["metadata"]["net_cash_operating_values"] = years_values
            
            if "net income" in table_markdown.lower():
                financial_line_items.append("net_income")
                table_markdown_doc["metadata"]["contains_net_income"] = True
                
            if "revenue" in table_markdown.lower():
                financial_line_items.append("revenue")
                table_markdown_doc["metadata"]["contains_revenue"] = True
                
            if "balance sheet" in table_markdown.lower():
                financial_line_items.append("balance_sheet")
                table_markdown_doc["metadata"]["contains_balance_sheet"] = True
                
            if financial_line_items:
                table_markdown_doc["metadata"]["financial_line_items"] = ",".join(financial_line_items)
            
            # Check if this is a cash flow related table
            if any(term in table_markdown.lower() for term in [
                "cash flow", "operating activities", "investing activities", 
                "financing activities", "net cash", "cash provided", "cash used"
            ]):
                table_markdown_doc["metadata"]["cash_flow_related"] = True
                
                # Check for specific year columns
                year_pattern = r"\b(19\d{2}|20\d{2})\b"
                years_mentioned = re.findall(year_pattern, table_markdown)
                if years_mentioned:
                    table_markdown_doc["metadata"]["years_mentioned"] = ",".join(sorted(set(years_mentioned)))
            
            # Check if this is a working capital related table
            if any(term in table_markdown.lower() for term in [
                "working capital", "accounts receivable", "inventory", "inventories",
                "accounts payable", "current assets", "current liabilities",
                "trade receivables", "trade payables", "net working capital"
            ]):
                table_markdown_doc["metadata"]["working_capital_related"] = True
                
                # Check for specific year columns
                year_pattern = r"\b(19\d{2}|20\d{2})\b"
                years_mentioned = re.findall(year_pattern, table_markdown)
                if years_mentioned:
                    table_markdown_doc["metadata"]["years_mentioned"] = ",".join(sorted(set(years_mentioned)))
            
            processed_docs.append(table_markdown_doc)
        
        if table_data:
            # Convert table_data (array) to markdown for text representation
            table_data_str = json.dumps(table_data, indent=2)
            table_data_doc = {
                "id": f"{doc_id}_table_data",
                "content": table_data_str,
                "metadata": {
                    **enhanced_metadata,
                    "type": "table_data",
                    "raw_table": table_data  # Store the original table structure
                }
            }
            processed_docs.append(table_data_doc)
        
        # Process post_text if available
        post_text = doc.get('post_text', '')
        if post_text:
            post_text_doc = {
                "id": f"{doc_id}_post_text",
                "content": post_text,
                "metadata": {
                    **enhanced_metadata,
                    "type": "post_text"
                }
            }
            processed_docs.append(post_text_doc)
    
    logger.info(f"Preprocessed {len(data)} documents into {len(processed_docs)} chunks")
    return processed_docs 