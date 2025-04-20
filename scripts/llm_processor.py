import json
import requests
import time
import os
import re
import hashlib
from pathlib import Path
import threading
import concurrent.futures
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

class CVInfoExtractor:
    # === Core Settings ===
    MODELS = ['llama3:latest', 'phi:latest', 'mistral:latest']  # Models to try in order
    INPUT_FILE = Path('./input.txt')  # Change this to your input file path
    OUTPUT_FILE = Path('./output.json')  # Change this to your output file path
    CACHE_DIR = Path('./cache')
    REQUEST_TIMEOUT = 30  # Switch to OpenRouter after this many seconds
    OLLAMA_API = "http://localhost:11434/api/generate"
    CONNECTION_RETRIES = 2  # Number of retries for local model
    CONTEXT_LENGTH = 8192  # Context length
    
    # OpenRouter settings
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    # OpenRouter models (free tier)
    OPENROUTER_MODELS = [
        'mistralai/mistral-7b-instruct:free',
        'shisa-ai/shisa-v2-llama3.3-70b:free',
        'cognitivecomputations/dolphin3.0-mistral-24b:free'
    ]
    
    # Prompt for extraction
    EXTRACTION_PROMPT = """You are a CV information extraction expert. Extract structured information from this text and format it as valid JSON.

IMPORTANT: Your response MUST be ONLY a valid JSON object with no additional text, explanation, or markdown formatting.

Extract these fields:
- Name: Full name of the person
- Email: Email address 
- Phone: Phone number with country code if available
- Address: Physical address
- Education: All education entries as an array of objects
- Experience: All work experience as an array of objects
- Skills: All skills as an array of strings

JSON Structure:
{
  "Name": "Person's full name",
  "Email": "email@example.com",
  "Phone": "+1 234 567 8900",
  "Address": "123 Example Street, Example City",
  "Education": [
    {
      "degree": "Degree name",
      "institution": "Institution name",
      "year": "Time period",
      "description": ["Achievement or detail 1", "Achievement or detail 2"]
    }
  ],
  "Experience": [
    {
      "job_title": "Position title",
      "company": "Company name",
      "duration": "Time period",
      "description": ["Responsibility or achievement 1", "Responsibility or achievement 2"]
    }
  ],
  "Skills": ["Skill 1", "Skill 2", "Skill 3"],
}

CV Text:
{cv_text}"""
    
    def __init__(self):
        self.cache_dir = self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def check_ollama_available(self):
        """Verify Ollama API is running"""
        try:
            r = requests.get("http://localhost:11434/api/version", timeout=5)
            r.raise_for_status()
            print(f"✅ Ollama is running (version: {r.json().get('version', 'unknown')})")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama is not available: {e}")
            return False
    
    def check_openrouter_available(self):
        """Verify OpenRouter API is available"""
        if not self.OPENROUTER_API_KEY:
            print("❌ OpenRouter API key is not configured")
            return False
        
        try:
            r = requests.get(
                f"{self.OPENROUTER_BASE_URL}/auth/key",
                headers={"Authorization": f"Bearer {self.OPENROUTER_API_KEY}"},
                timeout=5
            )
            r.raise_for_status()
            print("✅ OpenRouter API connection successful")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ OpenRouter API is not available: {e}")
            return False
    
    def get_cache_key(self, cv_text, model):
        """Generate a cache key based on model and CV text"""
        content = f"{model}:{cv_text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_from_cache(self, cache_key):
        """Get result from cache if available"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    print("📂 Using cached result")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Cache error: {e}")
                return None
        return None
    
    def save_to_cache(self, cache_key, result):
        """Save result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                print("📥 Result cached successfully")
        except Exception as e:
            print(f"⚠️ Error saving to cache: {e}")
    
    def call_ollama_api(self, model, prompt):
        """Call Ollama API with retry logic"""
        print(f"🔄 Using Ollama model: {model}")
        
        # Check if model is available
        try:
            model_response = requests.get(f"http://localhost:11434/api/tags", timeout=5)
            local_models = [m["name"] for m in model_response.json().get("models", [])]
            
            if model not in local_models and not model.endswith(':latest'):
                base_model = model.split(':')[0]
                if f"{base_model}:latest" not in local_models:
                    print(f"⚠️ Model '{model}' is not available locally.")
                    return None
        except requests.exceptions.RequestException:
            print("⚠️ Could not check available models. Proceeding anyway.")
        
        # Prepare payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.01,
                "num_predict": self.CONTEXT_LENGTH,
                "num_ctx": self.CONTEXT_LENGTH
            }
        }
        
        for attempt in range(self.CONNECTION_RETRIES + 1):
            try:
                print(f"🔄 API request attempt {attempt+1}...")
                response = requests.post(
                    self.OLLAMA_API, 
                    json=payload, 
                    timeout=self.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                print("✅ Ollama API request successful")
                return response.json().get("response", "")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Request error (attempt {attempt+1}/{self.CONNECTION_RETRIES+1}): {str(e)}")
                if attempt < self.CONNECTION_RETRIES:
                    backoff_time = 2 ** attempt
                    print(f"⏳ Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                    continue
                
                print("❌ All Ollama attempts failed")
                return None
    
    def call_openrouter_api(self, prompt):
        """Call OpenRouter API with fallback models"""
        if not self.OPENROUTER_API_KEY:
            print("❌ OpenRouter API key not configured")
            return None
            
        print("🌐 Falling back to OpenRouter API")
        
        # Try each OpenRouter model in sequence
        for model in self.OPENROUTER_MODELS:
            try:
                print(f"🔄 Using OpenRouter model: {model}")
                
                # Enhanced system message for JSON output
                system_message = """You are a CV information extractor that outputs ONLY valid JSON.
                IMPORTANT: Your entire response must be ONLY raw JSON with no markdown formatting, 
                no code blocks, and no explanatory text."""
                
                # Prepare the payload
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"}
                }
                
                # Make the API call
                headers = {
                    "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://cv-extractor.app"
                }
                
                print(f"📡 Sending request to OpenRouter API...")
                response = requests.post(
                    f"{self.OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT
                )
                
                # Check for HTTP errors
                if response.status_code != 200:
                    print(f"⚠️ OpenRouter API returned status code {response.status_code}")
                    continue
                
                # Parse the response
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    response_text = response_data['choices'][0]['message']['content']
                    print(f"✅ OpenRouter API request successful")
                    return response_text
                else:
                    print(f"⚠️ OpenRouter API returned empty response")
                    continue
                    
            except requests.exceptions.RequestException as e:
                print(f"⚠️ OpenRouter API error with model {model}: {e}")
                continue
        
        # If we get here, all models failed
        print(f"❌ All OpenRouter models failed")
        return None
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from the response"""
        if not response:
            return {"error": "Empty response"}
        
        try:
            # Clean the response to extract JSON
            print("🔍 Extracting JSON from response...")
            
            # Find the first opening brace and last closing brace
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            
            if first_brace == -1 or last_brace == -1 or first_brace > last_brace:
                # Try to find code block markers
                json_start = response.find('```json')
                json_end = response.rfind('```')
                
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    # Extract JSON from code block
                    json_content = response[json_start + 7:json_end].strip()
                    parsed_result = json.loads(json_content)
                else:
                    return {"error": "Could not find valid JSON in response"}
            else:
                # Extract the JSON part
                json_str = response[first_brace:last_brace + 1]
                parsed_result = json.loads(json_str)
            
            # Convert to standard format if needed
            if isinstance(parsed_result, dict):
                # Normalize field names
                normalized = {}
                
                # Map common field variations to standard names
                field_mapping = {
                    'name': 'Name',
                    'fullname': 'Name',
                    'full_name': 'Name',
                    'email': 'Email',
                    'email_address': 'Email',
                    'phone': 'Phone',
                    'phone_number': 'Phone',
                    'address': 'Address',
                    'location': 'Address',
                    'education': 'Education',
                    'experience': 'Experience',
                    'work_experience': 'Experience',
                    'employment': 'Experience',
                    'skills': 'Skills',
                }
                
                # Normalize field names
                for key, value in parsed_result.items():
                    normalized_key = field_mapping.get(key.lower(), key)
                    normalized[normalized_key] = value
                
                parsed_result = normalized
                
                # Normalize date formats in experience section
                if 'Experience' in parsed_result and parsed_result['Experience']:
                    for exp in parsed_result['Experience']:
                        if isinstance(exp, dict) and 'duration' in exp and exp['duration']:
                            exp['duration'] = self.normalize_dates(exp['duration'])
                
                return parsed_result
            else:
                return {"error": "Invalid JSON structure"}
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {str(e)}")
            return {
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": response[:100] + "..." if len(response) > 100 else response
            }
        except Exception as e:
            print(f"❌ Error extracting JSON: {str(e)}")
            return {
                "error": f"JSON extraction error: {str(e)}",
                "raw_response": response[:100] + "..." if len(response) > 100 else response
            }
    
    def normalize_dates(self, text: str) -> str:
        """
        Normalize date ranges by replacing various dash/hyphen characters with standard hyphens.
        This helps with displaying dates properly in the UI.
        """
        if not text:
            return text
        
        # Replace various dash characters with standard hyphens
        replacements = [
            ('–', '-'),  # en dash
            ('—', '-'),  # em dash
            ('−', '-'),  # minus sign
            ('‐', '-'),  # hyphen
            ('‑', '-'),  # non-breaking hyphen
            ('⁃', '-'),  # hyphen bullet
            ('⁻', '-'),  # superscript minus
            ('₋', '-'),  # subscript minus
        ]
        
        result = text
        for old, new in replacements:
            result = result.replace(old, new)
        
        return result
    
    def smart_truncate(self, cv_text: str, max_length: int = 6000) -> str:
        """Truncate CV text while preserving important sections"""
        if len(cv_text) <= max_length:
            return cv_text
            
        # Find section headers
        sections = ["education", "experience", "skills", "work history"]
        section_positions = []
        
        for section in sections:
            for match in re.finditer(rf"\b{section}\b", cv_text, re.IGNORECASE):
                section_positions.append(match.start())
        
        section_positions.sort()
        
        if not section_positions:
            # Simple truncation if no sections found
            return cv_text[:max_length]
        
        # Keep beginning, important sections, and end
        start_text = cv_text[:max_length//3]
        
        # Find middle section
        mid_point = len(cv_text) // 2
        closest_section = min(section_positions, key=lambda x: abs(x - mid_point))
        middle_text = cv_text[max(0, closest_section - max_length//3):closest_section + max_length//3]
        
        # End section
        end_text = cv_text[-max_length//3:]
        
        return f"{start_text}\n[...]\n{middle_text}\n[...]\n{end_text}"
    
    def extract_from_cv(self, cv_text):
        """Extract structured information from CV text with timeout support"""
        # Truncate text if needed
        cv_text = self.smart_truncate(cv_text)
        
        try:
            # Format prompt with escaped curly braces in cv_text to prevent format errors
            # First, properly escape any existing curly braces in the CV text
            escaped_cv_text = cv_text.replace("{", "{{").replace("}", "}}")
            prompt = self.EXTRACTION_PROMPT.format(cv_text=escaped_cv_text)
        except KeyError as e:
            print(f"⚠️ Error formatting prompt: {e}")
            # Fallback approach: manually insert the CV text
            parts = self.EXTRACTION_PROMPT.split("{cv_text}")
            if len(parts) == 2:
                prompt = parts[0] + cv_text + parts[1]
            else:
                # Last resort
                prompt = f"""Extract structured information from this CV as JSON: {cv_text}
                Return ONLY a valid JSON object with Name, Email, Phone, Address, Education, Experience and Skills fields."""
        
        result = None
        
        # First try Ollama models with timeout
        ollama_available = self.check_ollama_available()
        if ollama_available:
            for model in self.MODELS:
                # Check cache first
                cache_key = self.get_cache_key(cv_text, model)
                cached_result = self.get_from_cache(cache_key)
                if cached_result:
                    return cached_result
                
                print(f"\n🔄 Trying model: {model}")
                
                # Process with timeout
                thread_result = {"response": None, "completed": False}
                
                def process_model():
                    thread_result["response"] = self.call_ollama_api(model, prompt)
                    thread_result["completed"] = True
                
                # Start processing in a separate thread
                process_thread = threading.Thread(target=process_model)
                process_thread.daemon = True
                process_thread.start()
                
                # Wait for completion or timeout
                wait_time = 0
                check_interval = 0.5
                while wait_time < self.REQUEST_TIMEOUT and not thread_result["completed"]:
                    time.sleep(check_interval)
                    wait_time += check_interval
                    
                if thread_result["completed"] and thread_result["response"]:
                    # Success! Parse the response
                    result = self.extract_json_from_response(thread_result["response"])
                    if "error" not in result:
                        # Cache successful result
                        self.save_to_cache(cache_key, result)
                        return result
                    else:
                        print(f"⚠️ Model {model} gave error: {result.get('error')}")
                        # Try next model
                        continue
                else:
                    print(f"⏳ Model {model} timed out after {wait_time} seconds")
                    # Try next model
                    continue
        
        # If Ollama failed or timed out, try OpenRouter
        print("\n🌐 All Ollama models failed or timed out, trying OpenRouter...")
        if self.check_openrouter_available():
            # Check OpenRouter cache
            cache_key = self.get_cache_key(cv_text, "openrouter")
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                return cached_result
                
            # Call OpenRouter API
            response = self.call_openrouter_api(prompt)
            if response:
                result = self.extract_json_from_response(response)
                if "error" not in result:
                    # Cache successful result
                    self.save_to_cache(cache_key, result)
                    return result
        
        # If all methods failed
        print("❌ All extraction methods failed")
        return {
            "error": "All extraction methods failed",
            "Name": "Extraction Failed",
            "Email": "",
            "Phone": "",
            "Address": "",
            "Education": [],
            "Experience": [],
            "Skills": []
       }
    
    def extract_from_file(self, input_path=None, output_path=None):
        """Extract information from a file"""
        input_path = input_path or self.INPUT_FILE
        output_path = output_path or self.OUTPUT_FILE
        
        print(f"📄 Reading from: {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                cv_text = f.read().strip()
                
            if not cv_text:
                print("❌ Input file is empty")
                return False
                
            print(f"📝 Extracted {len(cv_text)} characters from file")
            
            # Process the CV text
            result = self.extract_from_cv(cv_text)
            
            # Save result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Result saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return False

def main():
    print("🚀 Starting CV Information Extractor")
    
    # Get file paths from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extract structured information from CV text files")
    parser.add_argument("--input", "-i", help="Input text file path", type=str)
    parser.add_argument("--output", "-o", help="Output JSON file path", type=str)
    args = parser.parse_args()
    
    extractor = CVInfoExtractor()
    
    # Use command line args if provided, otherwise use default paths
    input_path = Path(args.input) if args.input else extractor.INPUT_FILE
    output_path = Path(args.output) if args.output else extractor.OUTPUT_FILE
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
        
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract information from the file
    extractor.extract_from_file(input_path, output_path)
    
    print("\n✅ Process complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Process interrupted by user")
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()