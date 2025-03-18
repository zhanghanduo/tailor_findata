import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
import logging
from llama_index.schema import TextNode
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Prompt template for numerical operations
NUMERICAL_OPERATION_TEMPLATE = PromptTemplate(
    """You are a financial analyst tasked with performing numerical operations on financial data.

Retrieved financial information:
{context}

User query: {query}

Analysis from query: {analysis}

Your task is to:
1. Extract the relevant numerical values from the retrieved information
2. Perform the requested calculation or comparison
3. Provide the result with appropriate units and context

Be precise and show your work. Remember that financial calculations must be accurate.

Result:"""
)

def extract_financial_metrics(text: str) -> Dict[str, float]:
    """
    Extract financial metrics from text, including handling for dollar amounts, 
    percentages, and other numerical values with labels.
    
    Args:
        text: Text containing financial metrics.
        
    Returns:
        Dictionary mapping metric names to values.
    """
    metrics = {}
    
    # Pattern for currency with labeled metric
    # Example: "revenue was $10.5 million" or "net income: $3.2 billion"
    currency_metric_pattern = r'([a-z\s]+(?:revenue|income|profit|loss|sales|earnings|assets|liabilities|equity|margin|ratio|expense|cost|dividend|yield|return|growth))(?:\s*(?:was|of|:|is|at|reached))?\s*(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(million|billion|thousand|m|b|k)?'
    
    # Pattern for percentage with labeled metric
    # Example: "growth rate of 12.5%" or "profit margin: 8.3%"
    percentage_metric_pattern = r'([a-z\s]+(?:rate|margin|percentage|ratio|growth|yield|return|increase|decrease))(?:\s*(?:was|of|:|is|at|reached))?\s*([\d,]+\.?\d*)\s*%'
    
    # Pattern for labeled numerical value
    # Example: "2,500 employees" or "inventory turnover: 5.3"
    labeled_value_pattern = r'([\d,]+\.?\d*)\s+([a-z\s]+(?:employees|customers|stores|units|years|months|days|ratio|index|score|rating))'
    
    # Extract currency metrics
    for match in re.finditer(currency_metric_pattern, text, re.IGNORECASE):
        metric_name = match.group(1).strip().lower()
        value_str = match.group(2).replace(',', '')
        multiplier = 1
        
        if match.group(3):
            multiplier_str = match.group(3).lower()
            if 'billion' in multiplier_str or 'b' == multiplier_str:
                multiplier = 1e9
            elif 'million' in multiplier_str or 'm' == multiplier_str:
                multiplier = 1e6
            elif 'thousand' in multiplier_str or 'k' == multiplier_str:
                multiplier = 1e3
        
        try:
            value = float(value_str) * multiplier
            metrics[metric_name] = value
        except ValueError:
            continue
    
    # Extract percentage metrics
    for match in re.finditer(percentage_metric_pattern, text, re.IGNORECASE):
        metric_name = match.group(1).strip().lower()
        value_str = match.group(2).replace(',', '')
        
        try:
            value = float(value_str) / 100  # Convert percentage to decimal
            metrics[f"{metric_name} (%)"] = value
        except ValueError:
            continue
    
    # Extract labeled numerical values
    for match in re.finditer(labeled_value_pattern, text, re.IGNORECASE):
        value_str = match.group(1).replace(',', '')
        metric_name = match.group(2).strip().lower()
        
        try:
            value = float(value_str)
            metrics[metric_name] = value
        except ValueError:
            continue
    
    return metrics

def extract_all_numbers(text: str) -> List[Dict[str, Any]]:
    """
    Extract all numbers from text with surrounding context.
    
    Args:
        text: Text to extract numbers from.
        
    Returns:
        List of dictionaries with number values and context.
    """
    # General pattern for numbers with optional currency symbols and units
    pattern = r'(?:(?:\$|USD)?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k|%)?)'
    
    results = []
    for match in re.finditer(pattern, text):
        value_str = match.group(1).replace(',', '')
        try:
            value = float(value_str)
            # Get surrounding context (10 chars before and after)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            # Check if this is a currency value
            is_currency = '$' in context[max(0, match.start()-start-1):match.end()-start+1]
            
            # Check if this is a percentage
            is_percentage = '%' in context[match.end()-start:min(match.end()-start+1, len(context))]
            
            # Check for multipliers
            multiplier = 1
            for mult_term, mult_value in [
                ('billion', 1e9), ('million', 1e6), ('thousand', 1e3),
                ('b', 1e9), ('m', 1e6), ('k', 1e3)
            ]:
                if mult_term in context[match.end()-start:min(match.end()-start+10, len(context))]:
                    multiplier = mult_value
                    break
            
            # Apply multiplier and normalize
            value = value * multiplier
            if is_percentage:
                value = value / 100
            
            results.append({
                'value': value,
                'is_currency': is_currency,
                'is_percentage': is_percentage,
                'context': context.strip(),
                'position': match.start()
            })
        except ValueError:
            continue
    
    return results

def extract_table_data(table_text: str) -> pd.DataFrame:
    """
    Extract tabular data from text representation of a table.
    
    Args:
        table_text: Text representation of a table.
        
    Returns:
        pandas DataFrame containing the extracted table data.
    """
    lines = [line.strip() for line in table_text.split('\n') if line.strip()]
    
    # Filter out separator lines (those containing only |, -, and spaces)
    content_lines = [line for line in lines if not re.match(r'^[\-\|\s]+$', line)]
    
    if not content_lines:
        return pd.DataFrame()
    
    # Process each line
    rows = []
    for line in content_lines:
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
    
    # Try to infer if first row is header
    if len(rows) > 1:
        # Check if first row contains primarily text and subsequent rows contain numbers
        first_row_numeric_count = sum(1 for cell in rows[0] if re.search(r'^\d+\.?\d*$', cell))
        second_row_numeric_count = sum(1 for cell in rows[1] if re.search(r'^\d+\.?\d*$', cell))
        
        if first_row_numeric_count < second_row_numeric_count:
            # First row likely header
            df = pd.DataFrame(rows[1:], columns=rows[0])
        else:
            df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(rows)
    
    # Try to convert numeric columns
    for col in df.columns:
        try:
            # Replace common currency and numeric formatting
            df[col] = df[col].str.replace('$', '').str.replace(',', '').str.replace('%', '')
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            continue
    
    return df

def perform_numerical_operations(nodes: List[TextNode], query: str, analysis: Dict[str, Any], 
                                llm: Optional[OpenAI] = None) -> Dict[str, Any]:
    """
    Perform numerical operations on retrieved nodes based on query analysis.
    
    Args:
        nodes: List of retrieved TextNode objects.
        query: User query.
        analysis: Analysis of numerical aspects of the query.
        llm: Optional LLM for complex operations.
        
    Returns:
        Dictionary with results of numerical operations.
    """
    results = {
        "query": query,
        "extracted_metrics": {},
        "extracted_numbers": [],
        "tables": [],
        "computed_results": {},
        "operation_steps": []
    }
    
    # Extract metrics and numbers from each node
    for node in nodes:
        text = node.text
        metrics = extract_financial_metrics(text)
        for metric, value in metrics.items():
            if metric not in results["extracted_metrics"]:
                results["extracted_metrics"][metric] = []
            results["extracted_metrics"][metric].append({
                "value": value,
                "source": node.metadata.get("source", "unknown"),
                "node_id": node.id_
            })
        
        # Extract all numbers with context
        numbers = extract_all_numbers(text)
        for number in numbers:
            number["node_id"] = node.id_
            number["source"] = node.metadata.get("source", "unknown")
            results["extracted_numbers"].append(number)
        
        # Handle tables
        if node.metadata.get("is_table", False):
            try:
                df = extract_table_data(text)
                if not df.empty:
                    results["tables"].append({
                        "dataframe": df.to_dict(),
                        "shape": df.shape,
                        "node_id": node.id_,
                        "source": node.metadata.get("source", "unknown")
                    })
            except Exception as e:
                logger.warning(f"Failed to process table: {str(e)}")
    
    # Perform operations if LLM is provided
    if llm and analysis.get("needs_numerical_computation", False):
        # Prepare context from extracted data
        context = "Extracted metrics:\n"
        for metric, instances in results["extracted_metrics"].items():
            values = [instance["value"] for instance in instances]
            context += f"- {metric}: {values}\n"
        
        context += "\nExtracted tables:\n"
        for i, table in enumerate(results["tables"]):
            context += f"Table {i+1} (shape: {table['shape']}):\n"
            try:
                df = pd.DataFrame.from_dict(table["dataframe"])
                context += df.to_string(index=False) + "\n\n"
            except:
                context += "Could not format table.\n\n"
        
        # Use LLM to perform calculations
        response = llm.complete(
            NUMERICAL_OPERATION_TEMPLATE.format(
                context=context,
                query=query,
                analysis=analysis.get("analysis", "")
            )
        )
        
        results["llm_computation_result"] = str(response).strip()
    
    # Try to perform basic operations without LLM
    operations = analysis.get("operations", [])
    metrics = analysis.get("financial_metrics", [])
    
    for metric in metrics:
        if metric in results["extracted_metrics"]:
            metric_values = [instance["value"] for instance in results["extracted_metrics"][metric]]
            
            # Calculate basic statistics
            results["computed_results"][f"{metric}_mean"] = np.mean(metric_values)
            results["computed_results"][f"{metric}_median"] = np.median(metric_values)
            results["computed_results"][f"{metric}_min"] = np.min(metric_values)
            results["computed_results"][f"{metric}_max"] = np.max(metric_values)
            
            results["operation_steps"].append(f"Calculated statistics for {metric}")
    
    # Handle time period comparisons if mentioned in analysis
    time_periods = analysis.get("time_periods", [])
    if time_periods and len(time_periods) >= 2:
        # This would require more sophisticated logic for real implementation
        results["operation_steps"].append(f"Time period comparison requested: {', '.join(time_periods)}")
    
    return results 