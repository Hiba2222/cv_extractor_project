import os  # Pour les opérations sur les fichiers et dossiers
import fitz  # PyMuPDF - Bibliothèque pour l'extraction de texte des PDF
from pdf2image import convert_from_path  # Conversion de PDF en images
import google.generativeai as genai  # Gemini API for advanced OCR
from PIL import Image  # Manipulation d'images
import tempfile
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ExtractFromPDF:
    def __init__(self, raw_folder=None, output_folder=None, poppler_path=None, api_key=None):
        """
        Initialize the PDF extractor 
        
        Parameters:
        - raw_folder: Directory containing PDFs to process
        - output_folder: Directory to save extracted text
        - poppler_path: Path to Poppler binaries (needed for pdf2image)
        - api_key: Google API key for Gemini (can be set via GOOGLE_API_KEY env var)
        """
        # Store directory paths
        self.raw_folder = raw_folder
        self.output_folder = output_folder
        self.poppler_path = poppler_path
        
        # Get API key from param or environment
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini OCR. Set it in your .env file or pass it as a parameter.")
            
        # Configure Gemini API with the provided key
        genai.configure(api_key=self.api_key)
        
        # Initialize Gemini 1.5 Flash model for OCR
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Create output directory if provided
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def is_text_based_pdf(pdf_path):
        """
        Determines if a PDF contains selectable text
        Returns True if at least one page contains text, False otherwise
        """
        try:
            doc = fitz.open(pdf_path)  # Open PDF with PyMuPDF
            for page in doc:  # Iterate through each page
                text = page.get_text().strip()  # Extract text
                if text:  # If text is found
                    return True
            return False  # No text found in the document
        except Exception as e:
            print(f"Error checking PDF text: {e}")
            return False

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text from a text-based PDF (not scanned)
        Uses PyMuPDF for direct extraction
        """
        try:
            doc = fitz.open(pdf_path)
            all_text = ""
            for page in doc:
                all_text += page.get_text("text")  # Extract text with "text" format
            doc.close()  # Close the document
            return all_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_from_scanned_pdf(self, pdf_path):
        """
        Extract text from a scanned PDF (image-based)
        Uses Gemini API as an advanced OCR solution
        """
        try:
            # Convert PDF to list of images
            pages = convert_from_path(
                pdf_path, 
                poppler_path=self.poppler_path,
                dpi=300,
                thread_count=4
            )
            
            all_text = ""
            
            # Process each page with Gemini OCR
            for i, page_image in enumerate(pages):
                # Instruction for the Gemini model
                prompt = "Extract all text from this scanned CV page, preserving formatting and structure. Include all detailed information visible in the image."
                
                # Call the Gemini API with the image and prompt
                response = self.gemini_model.generate_content([prompt, page_image])
                
                # Extract and format the text
                page_text = response.text.strip()
                all_text += f"\n\n--- Page {i+1} ---\n\n{page_text}"
                
                # Process max 3 pages to avoid API limits
                if i >= 2:
                    all_text += "\n\n[Additional pages not processed due to API limits]"
                    break
            
            return all_text.strip()
                
        except Exception as e:
            print(f"Error processing scanned PDF with Gemini: {e}")
            return f"Error extracting text: {str(e)}"

    def process_pdf(self, filename):
        """
        Process an individual PDF file
        Detects type and uses appropriate extraction method
        """
        pdf_path = os.path.join(self.raw_folder, filename)
        print(f"\nProcessing: {filename}")
        
        try:
            # Determine PDF type and appropriate extraction
            if self.is_text_based_pdf(pdf_path):
                print("--> Detected as text-based PDF.")
                extracted_text = self.extract_text_from_pdf(pdf_path)
            else:
                print("--> Detected as image-based PDF.")
                extracted_text = self.extract_from_scanned_pdf(pdf_path)
            
            # Save extracted text to a .txt file
            if self.output_folder:
                base_name = os.path.splitext(filename)[0]  # Filename without extension
                output_path = os.path.join(self.output_folder, f"{base_name}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                print(f"Saved extracted text to: {output_path}")
            
            return extracted_text
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None

    def process_all_pdfs(self):
        """
        Process all PDF files in the source directory
        """
        if not self.raw_folder:
            raise ValueError("No input folder specified")
            
        results = {}
        
        for filename in os.listdir(self.raw_folder):
            if filename.endswith(".pdf"):  # Only process PDF files
                text = self.process_pdf(filename)
                if text:
                    results[filename] = text
        
        return results

# Only run this code if the file is executed directly (not imported)
if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("\nERROR: Google API key is missing. Please add your API key to the .env file.")
        print("1. Open the .env file in the project root directory")
        print("2. Add your key: GOOGLE_API_KEY=your_api_key_here\n")
        exit(1)
        
    # === Configuration for direct execution ===
    raw_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "input")
    output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "output")
    poppler_path = os.environ.get("POPPLER_PATH")
    
    # Create directories if they don't exist
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize extractor with API key from environment
    try:
        extractor = ExtractFromPDF(raw_folder, output_folder, poppler_path, api_key)
        print(f"\nProcessing PDF files from: {raw_folder}")
        extractor.process_all_pdfs()
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)