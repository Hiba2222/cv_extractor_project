# CV Extractor Evaluation Report

## 📊 Executive Summary

Our evaluation of three LLM models (Llama3, Mistral, and Phi) for CV information extraction shows:

- **Llama3** is the top performer with an overall score of **81.2%**
- **Mistral** achieves a moderate score of **58.9%**
- **Phi** shows basic extraction capabilities with **52.4%**

All models demonstrate strengths in extracting basic fields (name, contact info), but struggle with complex fields (experience).

## 🧪 Evaluation Methodology

### Data
- **Dataset**: 5 CVs with manually annotated ground truth (CV1, CV2, CV3, CV5, CV6)
- **Fields**: Name, Email, Phone, Skills, Education, Experience
- **Metrics**: Field-specific scores (0-100%)
- **Overall Score**: Weighted average across all fields

### Scoring System
- **Exact Match**: 100% (perfect extraction)
- **Partial Match**: 0-100% (based on content similarity)
- **No Match**: 0% (complete miss or incorrect extraction)
- **Field Weighting**: Name (20%), Contact (10% each), Skills (20%), Education (20%), Experience (20%)

## 📈 Model Performance

### Overall Performance

| Model   | Score | Percentage | Rank | Consistency (StdDev) |
|---------|-------|------------|------|----------------------|
| Llama3  | 0.812 | 81.2%      | 1    | 1.9%                 |
| Mistral | 0.589 | 58.9%      | 2    | 3.2%                 |
| Phi     | 0.524 | 52.4%      | 3    | 7.5%                 |

### Field-Specific Performance

| Field      | Llama3  | Mistral | Phi    | Most Accurate | Hardest to Extract |
|------------|---------|---------|--------|---------------|-------------------|
| Name       | 100.0%  | 75.0%   | 38.0%  | Llama3        | Phi               |
| Email      | 100.0%  | 74.0%   | 74.0%  | Llama3        | Mistral/Phi       |
| Phone      | 98.3%   | 75.0%   | 75.0%  | Llama3        | Mistral/Phi       |
| Skills     | 74.4%   | 42.0%   | 38.0%  | Llama3        | Phi               |
| Education  | 91.9%   | 74.0%   | 74.0%  | Llama3        | Mistral/Phi       |
| Experience | 46.8%   | 36.0%   | 38.0%  | Llama3        | Mistral           |
| **Overall**| **81.2%**| **58.9%**| **52.4%**| **Llama3**| **Experience**   |

## 🔍 Key Insights

### Model Strengths
- **Llama3**: Excellent at structured fields (100% on name/email), consistent performance across all CVs
- **Mistral**: Good at basic contact information (~75%), moderate performance on skills
- **Phi**: Acceptable performance on contact and education fields (>70%)

### Model Weaknesses
- **All Models**: Struggle with complex experience sections (<50%)
- **Mistral & Phi**: Difficulty with semi-structured skills sections (<45%)
- **Phi**: Inconsistent name extraction (38%)

### Field Extraction Difficulty
1. **Experience** (Hardest): Complex structure, dates, achievements, company details
2. **Skills**: Varied formats, technical terminology, classification challenges
3. **Education**: Dates, degree names, institutions
4. **Contact Information** (Easiest): Well-structured, standard formats

## 📋 Error Analysis

### Common Error Patterns
- **Format Inconsistency**: Different output structures for the same field
- **Incomplete Extraction**: Missing items in lists (skills, education)
- **Date Formatting**: Inconsistent date formats in experience entries
- **Detail Omission**: Missing specific details in experience descriptions

### Sample Error Cases
```
Ground Truth: "Project Manager at ABC Tech (Jan 2019 - Dec 2021)"
Llama3: "Project Manager, ABC Tech, 2019-2021"
Mistral: "Project Manager at ABC (2019-2021)"
Phi: "ABC Tech - Project Manager"
```

## 🔮 Recommendations

1. **Model Selection**: Use Llama3 as the primary extraction model (81.2% overall accuracy)
2. **Hybrid Approach**: Consider a field-specific approach where Llama3 handles most fields
3. **Post-Processing**: Implement specialized post-processing for experience fields (all models <50%)
4. **Schema Enforcement**: Strict output schema to ensure consistent formatting
5. **Performance Monitoring**: Regular evaluation with expanding ground truth dataset

## 📉 Limitations

- **Sample Size**: Limited to 5 CVs in current evaluation (CV1, CV2, CV3, CV5, CV6)
- **CV Complexity**: Varying complexity levels between samples
- **Language Limitation**: Evaluated only on English-language CVs
- **Format Bias**: Primarily PDF-based evaluation

---

## Appendix: Evaluation Details

### Cross-Validation Results

| Model   | CV1    | CV2    | CV3    | CV5    | CV6    | Average | StdDev |
|---------|--------|--------|--------|--------|--------|---------|--------|
| Llama3  | 81.2%  | 83.7%  | 80.6%  | 79.7%  | 81.0%  | 81.2%   | 1.9%   |
| Mistral | 62.4%  | 57.8%  | 59.8%  | 55.4%  | 59.2%  | 58.9%   | 3.2%   |
| Phi     | 58.6%  | 48.9%  | 43.2%  | 58.9%  | 52.4%  | 52.4%   | 7.5%   |

### Visualization References

- Complete performance visualizations available in `evaluation_reports/`
- Key visualizations:
  - `model_comparison_by_field.png` - Bar chart of field performance
  - `model_radar_comparison.png` - Radar chart of overall capabilities
  - `performance_heatmap.png` - Heatmap of model-field performance 