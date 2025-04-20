import json
import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from difflib import SequenceMatcher

# Try to import seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed, will use matplotlib for visualizations")

class CVEvaluator:
    def __init__(self, ground_truth_path, results_dir, models=None, fields=None):
        """
        Initialize evaluator with ground truth data and results directory
        
        Args:
            ground_truth_path: Path to JSON file with labeled CV data
            results_dir: Directory containing model results
            models: List of model names to evaluate (defaults to ['llama3', 'mistral', 'phi'])
            fields: List of fields to evaluate (defaults to name, email, etc.)
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.results_dir = results_dir
        self.models = models or ['llama3', 'mistral', 'phi']
        self.fields = fields or ['name', 'email', 'phone', 'education', 'experience', 'skills']
        
        # Define field types for processing
        self.string_fields = ['name', 'email', 'phone']
        self.list_fields = ['skills']
        self.complex_fields = ['education', 'experience']
        
    def _load_ground_truth(self, path):
        """Load ground truth data from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found at {path}")
            return {}
    
    def _load_model_results(self, cv_id, model_name):
        """Load results for a specific CV and model"""
        model_file = os.path.join(self.results_dir, model_name, f"{cv_id}.json")
        if not os.path.exists(model_file):
            return None
        
        with open(model_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _string_similarity(self, a, b):
        """Calculate string similarity ratio using SequenceMatcher"""
        if not a or not b:
            return 0
        
        # Normalize strings
        a = a.lower().strip()
        b = b.lower().strip()
        
        # Exact match
        if a == b:
            return 1.0
        
        # Use SequenceMatcher for better partial matching
        return SequenceMatcher(None, a, b).ratio()
    
    def _string_match_score(self, pred, truth):
        """Enhanced string matching score with normalization"""
        if not pred or not truth:
            return 0
        
        # Normalize strings
        pred = pred.lower().strip()
        truth = truth.lower().strip()
        
        # Exact match
        if pred == truth:
            return 1.0
        
        # Strong similarity (>80%)
        similarity = self._string_similarity(pred, truth)
        if similarity > 0.8:
            return similarity
        
        # Partial match - contained in other string
        if pred in truth:
            return 0.7
        elif truth in pred:
            return 0.6
            
        # Weak similarity
        if similarity > 0.5:
            return 0.5
            
        return 0
    
    def _list_match_score(self, pred_list, truth_list):
        """Improved matching score for lists (skills)"""
        if not pred_list or not truth_list:
            return 0
            
        if not isinstance(pred_list, list):
            pred_list = [pred_list]
            
        if not isinstance(truth_list, list):
            truth_list = [truth_list]
        
        # Empty list check
        if len(pred_list) == 0 or len(truth_list) == 0:
            return 0
            
        # For simple lists like skills
        if isinstance(pred_list[0], str) and isinstance(truth_list[0], str):
            # Calculate similarity for each pair of items
            matches = 0
            best_match_scores = []
            
            for truth_item in truth_list:
                item_scores = [self._string_similarity(truth_item, pred_item) for pred_item in pred_list]
                best_match = max(item_scores) if item_scores else 0
                best_match_scores.append(best_match)
            
            # Average of best matches
            avg_match_score = sum(best_match_scores) / len(truth_list)
            
            # Penalize for length differences
            length_penalty = min(len(pred_list), len(truth_list)) / max(len(pred_list), len(truth_list))
            
            return avg_match_score * length_penalty
        
        return 0
    
    def _complex_field_score(self, pred_data, truth_data, field_type):
        """
        Evaluate complex structured fields like education and experience
        
        Args:
            pred_data: Prediction data for the field
            truth_data: Ground truth data for the field
            field_type: Type of field ('education' or 'experience')
        """
        if not pred_data or not truth_data:
            return 0
            
        if not isinstance(pred_data, list):
            pred_data = [pred_data]
            
        if not isinstance(truth_data, list):
            truth_data = [truth_data]
            
        # Empty list check
        if len(pred_data) == 0 or len(truth_data) == 0:
            return 0
            
        # Education field has specific structure
        if field_type == 'education':
            # Expected keys for education entries
            keys = ['degree', 'institution', 'year']
            
            # Score each truth entry against best matching prediction
            entry_scores = []
            
            for truth_entry in truth_data:
                best_entry_score = 0
                
                for pred_entry in pred_data:
                    # Score individual fields in the entry
                    field_scores = {}
                    
                    for key in keys:
                        truth_value = truth_entry.get(key, '')
                        pred_value = pred_entry.get(key, '')
                        
                        if isinstance(truth_value, str) and isinstance(pred_value, str):
                            field_scores[key] = self._string_match_score(pred_value, truth_value)
                        else:
                            field_scores[key] = 0
                    
                    # Weighted average of field scores
                    entry_score = (field_scores.get('degree', 0) * 0.4 + 
                                  field_scores.get('institution', 0) * 0.4 + 
                                  field_scores.get('year', 0) * 0.2)
                    
                    if entry_score > best_entry_score:
                        best_entry_score = entry_score
                
                entry_scores.append(best_entry_score)
            
            # Average scores of all education entries
            return sum(entry_scores) / len(entry_scores) if entry_scores else 0
            
        # Experience field has different structure
        elif field_type == 'experience':
            # Expected keys for experience entries
            keys = ['title', 'company', 'duration', 'description']
            
            # Score each truth entry against best matching prediction
            entry_scores = []
            
            for truth_entry in truth_data:
                best_entry_score = 0
                
                for pred_entry in pred_data:
                    # Score individual fields in the entry
                    field_scores = {}
                    
                    for key in keys:
                        truth_value = truth_entry.get(key, '')
                        pred_value = pred_entry.get(key, '')
                        
                        if isinstance(truth_value, str) and isinstance(pred_value, str):
                            field_scores[key] = self._string_match_score(pred_value, truth_value)
                        else:
                            field_scores[key] = 0
                    
                    # Weighted average of field scores
                    entry_score = (field_scores.get('title', 0) * 0.35 + 
                                  field_scores.get('company', 0) * 0.35 + 
                                  field_scores.get('duration', 0) * 0.15 +
                                  field_scores.get('description', 0) * 0.15)
                    
                    if entry_score > best_entry_score:
                        best_entry_score = entry_score
                
                entry_scores.append(best_entry_score)
            
            # Average scores of all experience entries
            return sum(entry_scores) / len(entry_scores) if entry_scores else 0
            
        return 0
    
    def evaluate_cv(self, cv_id, model_results=None):
        """
        Evaluate a single CV's extraction results against ground truth
        
        Args:
            cv_id: ID of the CV to evaluate
            model_results: Dictionary of model results for this CV (optional)
            
        Returns:
            Dictionary with scores for each field and model
        """
        ground_truth = self.ground_truth.get(cv_id, {})
        if not ground_truth:
            print(f"Warning: No ground truth data found for CV ID: {cv_id}")
            return None
            
        scores = {}
        
        # Load model results if not provided
        if model_results is None:
            model_results = {}
            for model in self.models:
                result = self._load_model_results(cv_id, model)
                if result:
                    model_results[model] = result
            
            if not model_results:
                print(f"Warning: No model results found for CV ID: {cv_id}")
                return None
        
        for model, result in model_results.items():
            scores[model] = {}
            
            # Evaluate simple string fields
            for field in self.string_fields:
                if field in self.fields:
                    pred = result.get(field, '')
                    truth = ground_truth.get(field, '')
                    scores[model][field] = self._string_match_score(pred, truth)
            
            # Evaluate list fields
            for field in self.list_fields:
                if field in self.fields:
                    pred_list = result.get(field, [])
                    truth_list = ground_truth.get(field, [])
                    scores[model][field] = self._list_match_score(pred_list, truth_list)
                    
            # Evaluate complex fields
            for field in self.complex_fields:
                if field in self.fields:
                    pred_data = result.get(field, [])
                    truth_data = ground_truth.get(field, [])
                    scores[model][field] = self._complex_field_score(pred_data, truth_data, field)
                    
            # Overall score (weighted average)
            field_weights = {
                'name': 0.15,
                'email': 0.10,
                'phone': 0.10,
                'skills': 0.25,
                'education': 0.20,
                'experience': 0.20
            }
            
            # Calculate weighted average for overall score
            weighted_sum = 0
            weight_sum = 0
            
            for field in self.fields:
                if field in scores[model] and field in field_weights:
                    weighted_sum += scores[model][field] * field_weights[field]
                    weight_sum += field_weights[field]
            
            scores[model]['overall'] = weighted_sum / weight_sum if weight_sum > 0 else 0
            
        return scores
    
    def evaluate_all(self):
        """
        Evaluate all CVs in the ground truth data
        
        Returns:
            Dictionary with evaluation results for all CVs
        """
        all_results = {}
        
        for cv_id in self.ground_truth.keys():
            result = self.evaluate_cv(cv_id)
            if result:
                all_results[cv_id] = result
                
        return all_results
    
    def generate_report(self, evaluation_results):
        """
        Generate metrics and charts from evaluation results
        
        Args:
            evaluation_results: Dictionary of scores by CV and model
            
        Returns:
            Pandas DataFrame with results and saves charts
        """
        # Aggregate scores across all CVs
        agg_scores = {model: {field: 0 for field in self.fields + ['overall']} for model in self.models}
        cv_count = len(evaluation_results)
        
        field_scores_by_model = {model: {field: [] for field in self.fields + ['overall']} for model in self.models}
        
        for cv_id, cv_result in evaluation_results.items():
            for model in self.models:
                if model in cv_result:
                    for field in self.fields + ['overall']:
                        if field in cv_result[model]:
                            score = cv_result[model][field]
                            agg_scores[model][field] += score / cv_count
                            field_scores_by_model[model][field].append(score)
        
        # Create DataFrame for average scores
        df_avg = pd.DataFrame({
            model: {field: agg_scores[model][field] for field in self.fields + ['overall']}
            for model in self.models
        })
        
        # Create a separate DataFrame for standard deviations
        df_std = pd.DataFrame({
            model: {
                field: np.std(field_scores_by_model[model][field]) if field_scores_by_model[model][field] else 0
                for field in self.fields + ['overall']
            }
            for model in self.models
        })
        
        # Generate charts
        self._generate_charts(df_avg, df_std)
        
        # Generate per-model reports
        for model in self.models:
            self._generate_model_report(model, field_scores_by_model[model])
        
        return df_avg
    
    def _generate_model_report(self, model, field_scores):
        """Generate detailed report for a single model"""
        # Create output directory
        os.makedirs('evaluation_reports', exist_ok=True)
        
        # Create distributions data
        data = []
        for field in self.fields + ['overall']:
            for score in field_scores[field]:
                data.append({'Field': field, 'Score': score})
        
        if not data:
            return
            
        df = pd.DataFrame(data)
        
        # Create distribution plot
        plt.figure(figsize=(12, 8))
        
        if HAS_SEABORN:
            sns.boxplot(x='Field', y='Score', data=df)
        else:
            # Fallback to matplotlib boxplot
            fields = self.fields + ['overall']
            scores_by_field = [df[df['Field'] == field]['Score'].values for field in fields]
            plt.boxplot(scores_by_field, labels=fields)
            
        plt.title(f'{model} - Score Distribution by Field')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'evaluation_reports/{model}_score_distribution.png')
        plt.close()
        
        # Create summary statistics
        stats = df.groupby('Field')['Score'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        stats.to_csv(f'evaluation_reports/{model}_statistics.csv', index=False)
    
    def _generate_charts(self, df_avg, df_std):
        """Generate comparison charts"""
        # Create output directory
        os.makedirs('evaluation_reports', exist_ok=True)
        
        # Bar chart with error bars
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        field_names = self.fields + ['overall']
        x = np.arange(len(field_names))
        width = 0.2
        offsets = np.linspace(-width, width, len(self.models))
        
        # Create grouped bar chart
        for i, model in enumerate(self.models):
            model_scores = [df_avg[model][field] for field in field_names]
            model_errors = [df_std[model][field] for field in field_names]
            
            plt.bar(x + offsets[i], model_scores, width, label=model, alpha=0.7)
            plt.errorbar(x + offsets[i], model_scores, yerr=model_errors, fmt='none', capsize=5, ecolor='black', alpha=0.5)
        
        plt.xlabel('Fields')
        plt.ylabel('Score')
        plt.title('LLM Model Performance Comparison by Field')
        plt.xticks(x, field_names)
        plt.legend(title='Models')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('evaluation_reports/model_comparison_by_field.png')
        plt.close()
        
        # Heatmap of performance
        plt.figure(figsize=(12, 8))
        heatmap_data = df_avg.T
        
        if HAS_SEABORN:
            sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.2f')
        else:
            # Fallback to matplotlib for heatmap
            plt.imshow(heatmap_data.values, cmap='YlGnBu', vmin=0, vmax=1)
            plt.colorbar(label='Score')
            
            # Add text annotations
            for i in range(heatmap_data.shape[0]):
                for j in range(heatmap_data.shape[1]):
                    plt.text(j, i, f"{heatmap_data.values[i, j]:.2f}", 
                             ha="center", va="center", color="black")
            
            plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
            plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
            
        plt.title('Model Performance Heatmap')
        plt.tight_layout()
        plt.savefig('evaluation_reports/model_performance_heatmap.png')
        plt.close()
        
        # Radar chart for overall performance
        self._generate_radar_chart(df_avg)
        
        # Line plot for model comparison
        plt.figure(figsize=(12, 6))
        df_avg.loc[self.fields].T.plot(marker='o')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Performance Across Fields')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('evaluation_reports/model_field_comparison.png')
        plt.close()
    
    def _generate_radar_chart(self, df):
        """Generate radar chart for model comparison"""
        # Get data for radar chart
        categories = self.fields
        N = len(categories)
        
        # Create angles for each variable
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create radar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add each model to the chart
        for model in self.models:
            values = [df[model][field] for field in categories]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        plt.xticks(angles[:-1], categories)
        ax.set_ylim(0, 1)
        plt.title('Model Comparison by Field')
        plt.legend(loc='upper right')
        
        plt.savefig('evaluation_reports/model_radar_comparison.png')
        plt.close()

def main(ground_truth_path, results_dir, models=None):
    """
    Main function to run evaluation
    
    Args:
        ground_truth_path: Path to ground truth data
        results_dir: Directory containing model results
        models: List of models to evaluate (optional)
    
    Returns:
        DataFrame with evaluation results
    """
    evaluator = CVEvaluator(ground_truth_path, results_dir, models)
    results = evaluator.evaluate_all()
    report = evaluator.generate_report(results)
    
    # Save results to CSV
    report.to_csv('evaluation_reports/evaluation_results.csv')
    
    # Print summary
    print("\nEvaluation Summary:")
    print("===================")
    print(f"Total CVs evaluated: {len(results)}")
    print(f"Models compared: {', '.join(evaluator.models)}")
    print("\nOverall Scores:")
    for model in evaluator.models:
        print(f"- {model}: {report[model]['overall']:.3f}")
    
    print("\nDetailed results saved to 'evaluation_reports' directory")
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CV extraction models')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth JSON file')
    parser.add_argument('--results_dir', required=True, help='Directory containing model results')
    parser.add_argument('--models', nargs='+', help='List of models to evaluate')
    
    args = parser.parse_args()
    
    main(args.ground_truth, args.results_dir, args.models) 