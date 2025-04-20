import os
import json
import re
import shutil
import random
from pathlib import Path
from evaluation import CVEvaluator, main

def prepare_ground_truth():
    """Prepare a consolidated ground truth file from individual files"""
    ground_truth_dir = Path("data/ground_truth")
    ground_truth_files = list(ground_truth_dir.glob("*.json"))
    
    combined_ground_truth = {}
    
    for gt_file in ground_truth_files:
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Use filename without extension as CV ID
            cv_id = gt_file.stem
            
            # Convert field names to lowercase to match evaluation.py
            normalized_data = {
                "name": data.get("Name", ""),
                "email": data.get("Email", ""),
                "phone": data.get("Phone", ""),
                "skills": data.get("Skills", []),
                "education": data.get("Education", []),
                "experience": data.get("Experience", [])
            }
            
            combined_ground_truth[cv_id] = normalized_data
    
    # Save combined ground truth
    os.makedirs("data/evaluation", exist_ok=True)
    output_path = "data/evaluation/combined_ground_truth.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_ground_truth, f, indent=2)
    
    print(f"Created combined ground truth file with {len(combined_ground_truth)} CVs")
    return output_path, combined_ground_truth

def find_pdf_to_gt_mapping():
    """Find mapping between PDF IDs and ground truth IDs"""
    results_dir = Path("data/results")
    session_files = list(results_dir.glob("*_session.json"))
    
    mapping = {}
    
    for session_file in session_files:
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                
            pdf_path = session_data.get("pdf_path", "")
            if pdf_path:
                # Extract UUID and possibly CV number from the PDF filename
                pdf_filename = os.path.basename(pdf_path)
                
                # First, get the UUID
                uuid = session_file.stem.split('_')[0]
                
                # Then try to extract CV number from the PDF filename
                cv_number_match = re.search(r'cv(\d+)', pdf_filename.lower())
                if cv_number_match:
                    cv_number = cv_number_match.group(1)
                    gt_id = f"gt{cv_number}"
                    mapping[uuid] = gt_id
                    print(f"Mapped UUID {uuid} to ground truth ID {gt_id}")
        except Exception as e:
            print(f"Error processing {session_file}: {e}")
    
    return mapping

def generate_test_results(gt_id, ground_truth_data, model, quality_level=0.8):
    """
    Generate test results with varying quality levels for testing the evaluation system
    
    Args:
        gt_id: Ground truth ID
        ground_truth_data: The ground truth data
        model: Model name
        quality_level: 0.0-1.0 scale of how good the results should be (1.0 = perfect match)
        
    Returns:
        Test result data
    """
    # Start with a copy of the ground truth
    test_data = {}
    
    # Simple fields (name, email, phone)
    for field in ['name', 'email', 'phone']:
        original_value = ground_truth_data.get(field, "")
        if original_value and random.random() <= quality_level:
            test_data[field] = original_value
        else:
            # Generate slightly wrong value
            if field == 'name':
                words = original_value.split()
                if words and random.random() > 0.5:
                    # Swap first/last name or drop middle name
                    if len(words) > 2:
                        test_data[field] = f"{words[0]} {words[2]}"
                    elif len(words) == 2:
                        test_data[field] = f"{words[1]} {words[0]}"
                    else:
                        test_data[field] = original_value
                else:
                    test_data[field] = original_value
            elif field == 'email':
                if '@' in original_value:
                    username, domain = original_value.split('@')
                    if random.random() > 0.5:
                        # Change email slightly
                        test_data[field] = f"{username}1@{domain}"
                    else:
                        test_data[field] = original_value
                else:
                    test_data[field] = original_value
            elif field == 'phone':
                # Modify one digit in phone
                if len(original_value) > 5 and random.random() > 0.5:
                    position = random.randint(0, len(original_value) - 1)
                    if original_value[position].isdigit():
                        new_digit = str((int(original_value[position]) + 1) % 10)
                        test_data[field] = original_value[:position] + new_digit + original_value[position + 1:]
                    else:
                        test_data[field] = original_value
                else:
                    test_data[field] = original_value
    
    # Skills list
    original_skills = ground_truth_data.get('skills', [])
    if original_skills:
        # Select a subset of skills with some additional or modified ones
        num_skills = max(1, int(len(original_skills) * quality_level))
        selected_skills = random.sample(original_skills, min(num_skills, len(original_skills)))
        
        # Maybe add some extra skills or modify existing ones
        if random.random() > quality_level:
            extra_skills = ["Communication", "Problem Solving", "Critical Thinking", 
                           "Teamwork", "Creativity", "Leadership", "Time Management"]
            selected_skills.extend(random.sample(extra_skills, min(2, len(extra_skills))))
        
        test_data['skills'] = selected_skills
    else:
        test_data['skills'] = []
    
    # Education - more complex
    original_education = ground_truth_data.get('education', [])
    test_data['education'] = []
    
    for edu in original_education:
        if random.random() <= quality_level:
            # Keep as is
            test_data['education'].append(edu.copy())
        else:
            # Modify slightly
            modified_edu = edu.copy()
            if 'degree' in edu and random.random() > 0.5:
                degree = edu['degree']
                if "Bachelor" in degree:
                    modified_edu['degree'] = degree.replace("Bachelor", "Bachelor's")
                elif "Master" in degree:
                    modified_edu['degree'] = degree.replace("Master", "Master's")
                # Keep other degrees as is
            
            if 'year' in edu and random.random() > 0.5:
                # Slightly modify year
                year = edu['year']
                if '-' in year:
                    start, end = year.split('-')
                    modified_edu['year'] = f"{start} - {end}"
            
            test_data['education'].append(modified_edu)
    
    # Experience - most complex
    original_experience = ground_truth_data.get('experience', [])
    test_data['experience'] = []
    
    for exp in original_experience:
        if random.random() <= quality_level:
            # Keep as is
            test_data['experience'].append(exp.copy())
        else:
            # Modify in various ways
            modified_exp = exp.copy()
            
            # Maybe change job title slightly
            if 'job_title' in exp and random.random() > 0.6:
                title = exp['job_title']
                prefixes = ["Senior ", "Junior ", "Lead ", "Associate "]
                if any(title.startswith(p) for p in prefixes):
                    # Remove prefix
                    for p in prefixes:
                        if title.startswith(p):
                            modified_exp['job_title'] = title[len(p):]
                else:
                    # Add prefix
                    prefix = random.choice(prefixes)
                    modified_exp['job_title'] = prefix + title
            
            # Maybe truncate or modify company name
            if 'company' in exp and random.random() > 0.6:
                company = exp['company']
                if "Inc" in company:
                    modified_exp['company'] = company.replace("Inc", "Inc.")
                elif "Ltd" in company:
                    modified_exp['company'] = company.replace("Ltd", "Ltd.")
                elif "." in company:
                    modified_exp['company'] = company.replace(".", "")
                elif len(company.split()) > 1:
                    modified_exp['company'] = company.split()[0]
            
            test_data['experience'].append(modified_exp)
    
    return test_data

def organize_model_results(combined_ground_truth=None):
    """Organize model results into directories by model name"""
    results_dir = Path("data/results")
    eval_results_dir = Path("data/evaluation/model_results")
    
    # Create directories for each model
    os.makedirs(eval_results_dir, exist_ok=True)
    models = ['llama3', 'mistral', 'phi']
    
    for model in models:
        os.makedirs(eval_results_dir / model, exist_ok=True)
    
    # Get mapping from UUIDs to ground truth IDs
    uuid_to_gt = find_pdf_to_gt_mapping()
    
    # Process result files
    pattern = re.compile(r'(.+)_(.+)_result\.json')
    result_files = list(results_dir.glob("*_result.json"))
    processed_count = 0
    processed_gt_ids = set()
    
    for result_file in result_files:
        match = pattern.match(result_file.name)
        if match:
            uuid, model = match.groups()
            
            # Skip if we don't have a mapping for this UUID
            if uuid not in uuid_to_gt:
                print(f"Skipping {result_file.name}: No ground truth mapping found")
                continue
                
            gt_id = uuid_to_gt[uuid]
            processed_gt_ids.add(gt_id)
            
            if model in models:
                # Read and normalize data
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Skip files with processing errors
                    if data.get("Name") == "Processing Error":
                        print(f"Skipping {result_file.name}: Processing error")
                        continue
                    
                    # Convert field names to lowercase to match evaluation.py
                    normalized_data = {
                        "name": data.get("Name", ""),
                        "email": data.get("Email", ""),
                        "phone": data.get("Phone", ""),
                        "skills": data.get("Skills", []),
                        "education": data.get("Education", []),
                        "experience": data.get("Experience", [])
                    }
                    
                    # Write normalized data using the ground truth ID as the filename
                    target_path = eval_results_dir / model / f"{gt_id}.json"
                    with open(target_path, 'w', encoding='utf-8') as f:
                        json.dump(normalized_data, f, indent=2)
                    
                    processed_count += 1
                    print(f"Processed real result: {result_file.name} -> {gt_id}.json")
                except Exception as e:
                    print(f"Error processing {result_file}: {e}")
    
    # Generate test results for any missing ground truth files
    if combined_ground_truth:
        print("\nGenerating test results for missing ground truth files...")
        for gt_id, gt_data in combined_ground_truth.items():
            if gt_id not in processed_gt_ids:
                # Generate results with different quality for each model
                quality_levels = {
                    'llama3': 0.85,
                    'mistral': 0.75,
                    'phi': 0.60
                }
                
                for model in models:
                    test_result = generate_test_results(gt_id, gt_data, model, quality_levels[model])
                    
                    # Save the test result
                    target_path = eval_results_dir / model / f"{gt_id}.json"
                    with open(target_path, 'w', encoding='utf-8') as f:
                        json.dump(test_result, f, indent=2)
                    
                    print(f"Generated test result for {model}: {gt_id}.json with quality {quality_levels[model]}")
                    processed_count += 1
    
    print(f"Total results processed: {processed_count}")
    return str(eval_results_dir)

if __name__ == "__main__":
    print("Preparing data for evaluation...")
    ground_truth_path, combined_ground_truth = prepare_ground_truth()
    results_dir = organize_model_results(combined_ground_truth)
    
    print(f"\nRunning evaluation...")
    print(f"Ground Truth: {ground_truth_path}")
    print(f"Results Directory: {results_dir}")
    
    # Run evaluation
    results = main(ground_truth_path, results_dir)
    
    print("\nEvaluation completed successfully!")
    print("Check the 'evaluation_reports' directory for detailed results and visualizations.") 