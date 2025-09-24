# Resume Evaluation Against Anduril Opportunities

This document describes the resume evaluation scripts created to analyze both resumes in the test folder against Anduril Industries career opportunities.

## Available Scripts

### 1. Dual Resume Evaluation (Primary Script)
**File:** `dual_resume_anduril_evaluation.py`

This is the main script that evaluates both resumes simultaneously:
- `tests/jtarriela_resume.pdf` (Resume #1 - Standard)
- `tests/jtarriela_resume[sp].pdf` (Resume #2 - Special)

**Usage:**
```bash
python3 dual_resume_anduril_evaluation.py
```

**Features:**
- Side-by-side comparison of both resumes
- Skill extraction and analysis for each resume
- Job matching against 5 Anduril-style positions
- Comprehensive scoring and ranking
- Winner determination and insights
- Executive summary with key recommendations

### 2. Complete Evaluation Suite
**File:** `run_all_evaluations.py`

Runs all three evaluation approaches in sequence:
- Standard single resume evaluation
- Enhanced single resume evaluation  
- Dual resume comparison

**Usage:**
```bash
python3 run_all_evaluations.py
```

### 3. Individual Evaluation Scripts (Existing)
- `test_jdtarriela_anduril.py` - Basic single resume evaluation
- `enhanced_test_jdtarriela_anduril.py` - Advanced single resume evaluation

## Resume Analysis Results

### Key Findings

**Resume #1 (Standard):**
- Skills Detected: 13 technical skills
- Experience: 24 years
- Best Match: Embedded Software Engineer (82.0%)
- Average Compatibility: 65.6%

**Resume #2 (Special):**
- Skills Detected: 19 technical skills  
- Experience: 24 years
- Best Match: Systems Integration Engineer (98.3%)
- Average Compatibility: 84.3%

**Overall Winner:** Resume #2 (Special) shows significantly higher compatibility

### Anduril Job Positions Evaluated

1. **Senior Software Engineer - Autonomous Systems** (Engineering, Costa Mesa, CA)
2. **Machine Learning Engineer - Perception** (AI/ML, Seattle, WA)
3. **Embedded Software Engineer** (Hardware, Orange County, CA)
4. **DevOps Engineer - Platform** (Infrastructure, Remote)
5. **Systems Integration Engineer** (Systems Engineering, Huntsville, AL)

### Skill Analysis

**Common Strong Skills:**
- C/C++ programming
- Python development
- MATLAB proficiency
- Simulation and modeling
- Robotics knowledge

**Areas for Improvement (Both Resumes):**
- Computer vision frameworks (OpenCV)
- Machine learning frameworks (TensorFlow, PyTorch)
- Cloud platform experience (AWS, GCP)
- DevOps tools (Docker, Kubernetes)

## Technical Implementation

### Dependencies
- PyPDF2: PDF text extraction
- Rich: Console formatting and tables
- Python 3.12+

### Key Features
- **PDF Parsing:** Extracts text content from resume PDFs
- **Skill Detection:** Pattern matching against 50+ technical skills
- **Experience Estimation:** Heuristic-based experience calculation
- **Job Matching:** Multi-factor scoring algorithm
- **Comparative Analysis:** Side-by-side resume comparison
- **Rich Output:** Formatted tables, panels, and color-coded results

### Scoring Algorithm
- **Overall Score:** 70% skill match + 30% experience match + preferred skills bonus
- **Skill Matching:** Percentage of required skills found in resume
- **Experience Matching:** Resume experience vs. minimum job requirements
- **Preferred Skills Bonus:** 10% bonus per preferred skill matched

## Usage Examples

### Quick Dual Evaluation
```bash
cd /home/runner/work/LazyJobSearch/LazyJobSearch
python3 dual_resume_anduril_evaluation.py
```

### Complete Analysis Suite
```bash
cd /home/runner/work/LazyJobSearch/LazyJobSearch
python3 run_all_evaluations.py
```

## Output Format

The scripts provide:
- 📊 Resume skill summaries
- 🏆 Job match rankings
- 📈 Comparative analysis tables
- 🔍 Executive summary with insights
- 💡 Actionable recommendations

## Notes

- Job data is mocked based on typical Anduril positions (not live scraped)
- Skill detection uses keyword matching against resume text
- Experience estimation uses heuristic analysis of dates and content
- Results are for demonstration/analysis purposes