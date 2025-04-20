#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test CV Info Extractor
---------------------
This script tests the CVInfoExtractor class with a sample CV, focusing only on prompt formatting.

Usage:
    python scripts/test_extractor.py
"""

import os
import json
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the CVInfoExtractor
from scripts.llm_processor import CVInfoExtractor

# Load environment variables
load_dotenv()

# Configure standard output to use UTF-8 encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Sample CV text
SAMPLE_CV = """
John Smith
Email: john.smith@example.com
Phone: (123) 456-7890

EDUCATION
---------
University of Example
Bachelor of Science in Computer Science
2016 - 2020
- GPA: 3.8/4.0
- Dean's List all semesters

EXPERIENCE
----------
Software Engineer
Example Corp
January 2020 - Present
- Developed and maintained web applications using React and Node.js
- Implemented RESTful APIs and microservices
- Collaborated with cross-functional teams to deliver high-quality software

Intern
Tech Startup Inc.
Summer 2019
- Assisted in developing a mobile application using Flutter
- Participated in daily stand-up meetings and sprint planning
- Gained experience with Agile development methodologies

SKILLS
------
- Programming Languages: Python, JavaScript, Java, C++
- Web Technologies: HTML, CSS, React, Node.js
- Database Systems: MySQL, MongoDB
- Tools: Git, Docker, Jenkins, AWS

"""

def print_result(result):
    """Pretty print the extraction result with improved Unicode handling"""
    try:
        if "error" in result:
            print(f"❌ Error: {result.get('error')}")
            return
            
        print("\n✅ Extraction Result:\n")
        print(f"📝 Name: {result.get('Name', 'Not found')}")
        print(f"📧 Email: {result.get('Email', 'Not found')}")
        print(f"📞 Phone: {result.get('Phone', 'Not found')}")
        print(f"🏠 Address: {result.get('Address', 'Not found')}")
        
        print("\n📚 Education:")
        for edu in result.get('Education', []):
            print(f"  • {edu.get('degree')} at {edu.get('institution')} ({edu.get('year')})")
        
        print("\n💼 Experience:")
        for exp in result.get('Experience', []):
            print(f"  • {exp.get('job_title')} at {exp.get('company')} ({exp.get('duration')})")
        
        print("\n🔧 Skills:")
        for skill in result.get('Skills', []):
            print(f"  • {skill}")
            
        # Flush to ensure all content is displayed
        sys.stdout.flush()
    except UnicodeEncodeError as e:
        print(f"Unicode error in print_result: {e}")
        print("Trying to print with encoding fix...")
        print_raw_json(result)
    except Exception as e:
        print(f"Error in print_result: {str(e)}")
        traceback.print_exc()

def print_raw_json(result):
    """Fallback print method that outputs raw JSON with error handling"""
    try:
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        print("Raw JSON result:")
        for line in json_str.splitlines():
            # Handle each line individually to catch specific problematic characters
            try:
                print(line)
            except UnicodeEncodeError:
                # Replace problematic characters with '?'
                print(line.encode('ascii', 'replace').decode('ascii'))
    except Exception as e:
        print(f"Error in print_raw_json: {str(e)}")

def print_safe(text, lines_to_show=5):
    """Safely print text with Unicode characters"""
    try:
        all_lines = text.splitlines()
        
        # Print first n lines
        first_lines = all_lines[:lines_to_show]
        print(f"First {len(first_lines)} lines:")
        for line in first_lines:
            try:
                print(f"  {line}")
            except UnicodeEncodeError:
                print(f"  {line.encode('ascii', 'replace').decode('ascii')}")
            
        # Print last n lines
        last_lines = all_lines[-lines_to_show:]
        print(f"Last {len(last_lines)} lines:")
        for line in last_lines:
            try:
                print(f"  {line}")
            except UnicodeEncodeError:
                print(f"  {line.encode('ascii', 'replace').decode('ascii')}")
        
        # Flush to ensure all content is displayed
        sys.stdout.flush()
            
    except Exception as e:
        print(f"Error printing text: {str(e)}")
        traceback.print_exc()

def main():
    """Main function to test the CV extractor"""
    try:
        print("\n===== CV Info Extractor Test =====\n")
        
        # Initialize the extractor
        extractor = CVInfoExtractor()
        
        print("\n📄 Testing prompt formatting...\n")
        
        # Test the prompt formatting
        try:
            # Get the CV text ready
            cv_text = SAMPLE_CV.strip()
            
            # Test method 1: Direct string replacement
            print("Method 1: Direct string replacement")
            parts = extractor.EXTRACTION_PROMPT.split("{cv_text}")
            if len(parts) == 2:
                prompt = parts[0] + cv_text + parts[1]
                print("✅ Method 1 successful!")
                # Print parts safely
                print_safe(prompt)
            else:
                print("❌ Method 1 failed - couldn't find {cv_text} in prompt")
            
            print("\nMethod 2: Using safe formatting")
            # Test method 2: Replace with escaped braces
            try:
                escaped_cv_text = cv_text.replace("{", "{{").replace("}", "}}")
                prompt = extractor.EXTRACTION_PROMPT.format(cv_text=escaped_cv_text)
                print("✅ Method 2 successful!")
                # Print parts safely
                print_safe(prompt)
            except Exception as e:
                print(f"❌ Method 2 failed: {str(e)}")
            
            print("\nMethod 3: Simple approach")
            # Test method 3: Simplest approach
            try:
                simple_prompt = f"""Extract structured information from this CV as JSON:

{cv_text}

Return ONLY a valid JSON object with Name, Email, Phone, Address, Education, Experience and Skills fields."""
                
                print("✅ Method 3 successful!")
                # Print parts safely
                print_safe(simple_prompt)
            except Exception as e:
                print(f"❌ Method 3 failed: {str(e)}")
            
            print("\n📥 Prompt formatting tests completed")
                
        except Exception as e:
            print(f"❌ Error in prompt formatting test: {str(e)}")
            traceback.print_exc()
        
        print("\n===== Test Complete =====\n")
        # Final flush to ensure all output is displayed
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Fatal error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 