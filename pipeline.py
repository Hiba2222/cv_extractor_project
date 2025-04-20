"""
CV Extractor Pipeline
---------------------
This script runs the complete CV extraction pipeline:
1. Extract text from PDFs in the input folder
2. Process extracted text with various LLMs
3. Evaluate results against ground truth (if available)

Usage:
    python pipeline.py [--input INPUT_DIR] [--models MODEL1,MODEL2]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the scripts directory to the path
scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
sys.path.append(scripts_dir)

# Import the necessary modules
from scripts.pdf_extractor import ExtractFromPDF
from scripts.llm_processor import CVInfoExtractor

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CV Extractor Pipeline")
    parser.add_argument("--input", type=str, default="data/input",
                        help="Directory containing PDF files to process")
    parser.add_argument("--output", type=str, default="data/output",
                        help="Directory to save extracted text")
    parser.add_argument("--results", type=str, default="data/results",
                        help="Directory to save structured JSON results")
    parser.add_argument("--models", type=str, default="phi,mistral,llama3",
                        help="Comma-separated list of models to use")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate results against ground truth")
    
    return parser.parse_args()

def main():
    """Main pipeline function"""
    # Parse arguments
    args = parse_arguments()
    
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\nERROR: Google API key is missing. Please add your API key to the .env file.")
        print("1. Create a .env file in the project root directory")
        print("2. Add your key: GOOGLE_API_KEY=your_api_key_here\n")
        return 1
    
    # Create necessary directories
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    results_dir = os.path.abspath(args.results)
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if there are any PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"\nERROR: No PDF files found in {input_dir}")
        print(f"Please add PDF files to the input directory and try again.\n")
        return 1
    
    print("\n--- CV Extractor Pipeline ---\n")
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    # Step 1: Extract text from PDFs
    print("\n[1/2] Extracting text from PDFs...")
    poppler_path = os.environ.get("POPPLER_PATH")
    
    extractor = ExtractFromPDF(
        raw_folder=input_dir,
        output_folder=output_dir,
        poppler_path=poppler_path,
        api_key=api_key
    )
    
    extraction_results = extractor.process_all_pdfs()
    print(f"Extracted text from {len(extraction_results)} PDF files")
    
    # Step 2: Process the extracted text with LLMs
    print("\n[2/2] Processing extracted text with LLMs...")
    models = args.models.split(",")
    print(f"Using models: {', '.join(models)}")
    
    all_results = {}
    
    for model in models:
        print(f"\nProcessing with model: {model}...")
        processor = CVInfoExtractor()
        
        model_results = {}
        
        for filename, text in extraction_results.items():
            print(f"  Processing {filename}...")
            base_name = os.path.splitext(filename)[0]
            try:
                result = processor.extract_from_cv(text)
                model_results[filename] = result
                
                # Save individual result
                result_file = os.path.join(results_dir, f"{base_name}_{model}_result.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Error processing {filename} with {model}: {e}")
        
        all_results[model] = model_results
    
    # Save combined results
    combined_file = os.path.join(results_dir, "combined_results.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n--- Pipeline Complete ---")
    print(f"Extracted text files saved to: {output_dir}")
    print(f"Structured results saved to: {results_dir}")
    print(f"Combined results saved to: {combined_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 