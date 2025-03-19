import json
import os
from bs4 import BeautifulSoup


def format_html_table_for_llm(html_table):
    """Convert HTML table to a markdown-style text table that's easier for LLMs to understand"""
    if not html_table or html_table.strip() == '':
        return "No table data found."
        
    soup = BeautifulSoup(html_table, 'html.parser')
    
    # Extract all rows
    rows = soup.find_all('tr')
    if not rows:
        return "No table data found."
    
    # Process the table data
    table_data = []
    for row in rows:
        cells = row.find_all(['td', 'th'])
        # Skip the first cell if it's just a row number
        if cells and cells[0].text.strip().isdigit():
            row_data = [cell.text.strip() for cell in cells[1:]]
        else:
            row_data = [cell.text.strip() for cell in cells]
        table_data.append(row_data)
    
    # Format the table data as markdown
    return format_table_for_llm(table_data)


def format_table_for_llm(table_data):
    """
    Convert table data to a markdown-style text table that's easier for LLMs to understand.
    
    Args:
        table_data: List of lists representing table rows and columns
        
    Returns:
        Formatted table as a string
    """
    if not table_data or not isinstance(table_data, list) or len(table_data) == 0:
        return "No table data found."
    
    # Process the table data
    # Convert all values to strings and handle None values
    processed_table = []
    for row in table_data:
        if not isinstance(row, list):
            # Skip non-list rows
            continue
        processed_row = []
        for cell in row:
            if cell is None:
                processed_row.append("")
            else:
                processed_row.append(str(cell).strip())
        if processed_row:  # Only add non-empty rows
            processed_table.append(processed_row)
    
    if not processed_table:
        return "No valid table data found."
    
    # Ensure all rows have the same number of columns by padding with empty strings
    max_cols = max(len(row) for row in processed_table)
    padded_table = [row + [''] * (max_cols - len(row)) for row in processed_table]
    
    # Determine column widths for proper alignment
    col_widths = [max(len(row[i]) for row in padded_table) for i in range(max_cols)]
    
    # Create header row with column names
    header_row = padded_table[0]
    header_separator = ['-' * width for width in col_widths]
    
    # Format the table with proper alignment
    formatted_table = []
    formatted_table.append(' | '.join(header_row[i].ljust(col_widths[i]) for i in range(max_cols)))
    formatted_table.append(' | '.join(header_separator[i] for i in range(max_cols)))
    
    # Add data rows
    for row in padded_table[1:]:
        formatted_table.append(' | '.join(row[i].ljust(col_widths[i]) for i in range(max_cols)))
    
    return '\n'.join(formatted_table)


def transform_to_markdown(input_file, output_file):
    """
    Transform the ConvFinQA dataset into markdown format for RAG knowledge base
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output markdown file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(data):
            # Extract document ID
            doc_id = sample.get('id', f'document_{i}')
            
            # Extract context information
            pre_text = ""
            post_text = ""
            formatted_table = ""
            original_table_data = []
            
            # First check if the sample has the annotation field
            if "annotation" in sample:
                annos = sample['annotation']
                pre_text = annos.get('amt_pre_text', '')
                post_text = annos.get('amt_post_text', '')
                html_table = annos.get('amt_table', '').replace('<td></td>', '<td>-</td>')
                formatted_table = format_html_table_for_llm(html_table)
            
            # If annotation doesn't have the required fields or doesn't exist,
            # check if the sample has the direct fields
            if not pre_text and "pre_text" in sample:
                pre_text = ' '.join(sample.get("pre_text", []))
            
            if not post_text and "post_text" in sample:
                post_text = ' '.join(sample.get("post_text", []))
            
            if not formatted_table and "table" in sample:
                table_data = sample.get("table", [])
                formatted_table = format_table_for_llm(table_data)
                original_table_data = table_data
            elif not formatted_table and "table_ori" in sample:
                table_data = sample.get("table_ori", [])
                formatted_table = format_table_for_llm(table_data)
                original_table_data = table_data
            
            # Write to markdown file
            f.write(f'# {doc_id}\n\n')
            
            # Write full content first
            f.write('## Content\n\n')
            f.write(f'{pre_text}\n\n{formatted_table}\n\n{post_text}\n\n')
            
            # Write individual components
            if pre_text:
                f.write('## Pre-text\n\n')
                f.write(f'{pre_text}\n\n')
            
            if formatted_table:
                f.write('## Table (Markdown Format)\n\n')
                f.write(f'{formatted_table}\n\n')
            
            if original_table_data:
                f.write('## Table (Array Format)\n\n')
                f.write('```json\n')
                f.write(json.dumps(original_table_data, indent=2))
                f.write('\n```\n\n')
            
            if post_text:
                f.write('## Post-text\n\n')
                f.write(f'{post_text}\n\n')
            
            # Add separator between documents
            f.write('---\n\n')


def transform_to_json(input_file, output_file):
    """
    Transform the ConvFinQA dataset into JSON format for RAG knowledge base
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_data = []
    
    for i, sample in enumerate(data):
        # Extract document ID
        doc_id = sample.get('id', f'document_{i}')
        
        # Extract context information
        pre_text = ""
        post_text = ""
        formatted_table = ""
        original_table_data = []
        
        # First check if the sample has the annotation field
        # if "annotation" in sample:
        #     annos = sample['annotation']
        #     pre_text = annos.get('amt_pre_text', '')
        #     post_text = annos.get('amt_post_text', '')
        #     html_table = annos.get('amt_table', '').replace('<td></td>', '<td>-</td>')
        #     formatted_table = format_html_table_for_llm(html_table)
        
        # If annotation doesn't have the required fields or doesn't exist,
        # check if the sample has the direct fields
        if not pre_text and "pre_text" in sample:
            pre_text = ' '.join(sample.get("pre_text", []))
        
        if not post_text and "post_text" in sample:
            post_text = ' '.join(sample.get("post_text", []))
        
        if not formatted_table and "table" in sample:
            table_data = sample.get("table", [])
            formatted_table = format_table_for_llm(table_data)
            original_table_data = table_data
        elif not formatted_table and "table_ori" in sample:
            table_data = sample.get("table_ori", [])
            formatted_table = format_table_for_llm(table_data)
            original_table_data = table_data
        
        # Create document for RAG - modified structure with decoupled fields
        document = {
            "id": doc_id,
            "content": f"{pre_text}\n\n{formatted_table}\n\n{post_text}".strip(),
            "pre_text": pre_text,
            "post_text": post_text,
            "table_markdown": formatted_table,
            "table_data": original_table_data,
            "metadata": {
                "source": "ConvFinQA",
                "filename": sample.get("filename", "")
            }
        }
        
        output_data.append(document)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform ConvFinQA dataset for RAG knowledge base')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--format', type=str, choices=['json', 'markdown'], default='json',
                        help='Output format (json or markdown)')
    
    args = parser.parse_args()
    
    if args.format == 'json':
        transform_to_json(args.input, args.output)
    else:
        transform_to_markdown(args.input, args.output)
    
    print(f"Transformation complete. Output saved to {args.output}") 