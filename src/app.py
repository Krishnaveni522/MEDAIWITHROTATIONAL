"""
Flask application for Medical Instruction System
"""
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from utils.ocr_easyocr import extract_text_from_image
from utils.database import (
    init_database, get_all_drugs, get_all_drugs_dict, 
    add_drug, update_drug, delete_drug, load_from_docx_to_db
)
from difflib import SequenceMatcher
import re
import os

#app = Flask(__name__)
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'medai-secret-key-2025-change-in-production'

# Initialize database on startup
init_database()

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient')
def patient():
    return render_template('patient.html')

def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _snippet(text: str, max_sentences: int = None, max_chars: int = None) -> str:
    if not text:
        return ""
    # Return full text - no truncation for patient viewing
    return text.strip()

def _expand_variants(name: str) -> set:
    name = (name or "").strip()
    if not name:
        return set()
    variants = {name.lower()}
    if "/" in name:
        parts = [p.strip() for p in name.split("/")]
        for p in parts:
            if p:
                # add segment alone (e.g. "cream")
                variants.add(p.lower())
        # if the first segment has more than one word, treat last word as form and build "core form"
        first = parts[0]
        fw = first.split()
        core = " ".join(fw[:-1]) if len(fw) > 1 else fw[0]
        for p in parts:
            last = p.split()[-1] if p.split() else p
            variants.add(f"{core} {last}".strip().lower())
    return variants

# add a small set of generic form words to avoid matching only "cream/gel/ointment"
FORMS = {"cream", "gel", "ointment", "shampoo", "syrup", "tablet", "capsule", "solution", "ointment/cream", "ointment/gel"}

def _is_core_token_match(core_tokens, text_words):
    """Return True if any non-form core token appears in OCR words (exact token match)."""
    for t in core_tokens:
        if t and t in text_words:
            return True
    return False

def match_instructions(extracted_text: str, cutoff: float = 0.6, max_results: int = 10):
    """
    Return list of (drug_name, matched_variant, instruction_snippet, score).
    Enhanced to match all keywords from drug name in OCR text.
    """
    instructions_db = get_all_drugs_dict()
    
    if not extracted_text:
        return []

    text = extracted_text.lower()
    text_words = [w.strip(".,;:()[]/%\"'") for w in text.split() if w.strip()]
    text_join = " ".join(text_words)

    detected_forms = set(text_words) & FORMS
    detected_nonform = set(text_words) - FORMS

    results = []
    for drug, instr in instructions_db.items():
        if not drug or not drug.strip():
            continue

        drug_lower = drug.lower()
        drug_tokens = set([t.strip() for t in re.split(r"[\s/,]+", drug_lower) if t.strip()])
        
        # Remove common form words from drug tokens for core matching
        drug_core_tokens = drug_tokens - FORMS
        
        # If drug has no core tokens (only forms), use all tokens
        if not drug_core_tokens:
            drug_core_tokens = drug_tokens
        
        # Check if ALL core keywords from drug name appear in OCR text
        matched_tokens = drug_core_tokens & set(text_words)
        keyword_match_ratio = len(matched_tokens) / len(drug_core_tokens) if drug_core_tokens else 0
        
        # Require at least one core token match to be considered
        if not matched_tokens:
            continue

        variants = _expand_variants(drug)
        best_score = 0.0
        best_variant = None
        any_candidate = False

        for cand in variants:
            cand = cand.strip().lower()
            if not cand:
                continue

            cand_tokens = set([t for t in re.split(r"\s+", cand) if t])
            # skip bare form-only candidates
            if len(cand_tokens) == 1 and list(cand_tokens)[0] in FORMS:
                continue

            any_candidate = True

            # scoring - enhanced to favor complete keyword matches
            if cand in text_join:
                score = 1.0
            else:
                # Token overlap score
                overlap = 0.0
                if cand_tokens:
                    overlap = len(cand_tokens & set(text_words)) / len(cand_tokens)
                
                # Sequence matching score
                max_ngram_ratio = 0.0
                window_size = max(1, len(cand_tokens))
                for i in range(0, max(1, len(text_words) - window_size + 1)):
                    window = " ".join(text_words[i:i + window_size])
                    r = _seq_ratio(cand, window)
                    if r > max_ngram_ratio:
                        max_ngram_ratio = r
                whole_ratio = _seq_ratio(cand, text_join)
                max_ngram_ratio = max(max_ngram_ratio, whole_ratio)
                
                # Combine scores - give more weight to keyword matching
                score = max(max_ngram_ratio, overlap * 0.9, keyword_match_ratio * 0.85)

            if score > best_score:
                best_score = score
                best_variant = cand

            if best_score >= 0.999:
                break

        if not any_candidate:
            continue

        # Enhanced scoring based on keyword match ratio
        # Boost score if most/all keywords are matched
        final_score = best_score
        if keyword_match_ratio >= 1.0:  # All keywords matched
            final_score = max(final_score, 0.95)
        elif keyword_match_ratio >= 0.8:  # Most keywords matched
            final_score = max(final_score, 0.85)

        if final_score >= cutoff and matched_tokens:
            snippet = _snippet(instr)  # Return full instructions
            results.append((drug, best_variant or drug, snippet, round(final_score, 3)))

    results.sort(key=lambda x: (-x[3], x[0]))
    return results[:max_results]

@app.route('/patient/upload', methods=['POST'])
def patient_upload():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('patient'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('patient'))

    if file and allowed_file(file.filename):
        try:
            print(f"\n[UPLOAD] Processing file: {file.filename}", flush=True)
            extracted_text = extract_text_from_image(file) or ""
            print(f"[UPLOAD] OCR extracted {len(extracted_text)} characters", flush=True)
            print(f"[UPLOAD] OCR TEXT: {extracted_text[:200]}", flush=True)

            matches = match_instructions(extracted_text, cutoff=0.25, max_results=10)
            print(f"[UPLOAD] Found {len(matches)} matches", flush=True)

            return render_template('results.html', instructions=matches, ocr_text=extracted_text, search_mode='image')
        except Exception as e:
            print(f"[UPLOAD] ERROR: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('patient'))
    else:
        flash('Invalid file type. Please upload an image (JPG, PNG, JPEG)', 'error')
        return redirect(url_for('patient'))

@app.route('/patient/search', methods=['POST'])
def patient_search():
    search_text = request.form.get('search_text', '').strip()
    
    if not search_text:
        flash('Please enter a medicine name to search', 'error')
        return redirect(url_for('patient'))
    
    print("SEARCH TEXT:", search_text)
    
    # Use the same matching function for consistency
    matches = match_instructions(search_text, cutoff=0.25, max_results=10)
    
    return render_template('results.html', instructions=matches, ocr_text=search_text, search_mode='text')

@app.route('/admin')
def admin():
    """Admin view: list all medications from database."""
    medicines = get_all_drugs()
    message = request.args.get('message', '')
    return render_template('admin.html', medicines=medicines, message=message)

@app.route('/admin/add', methods=['POST'])
def admin_add():
    """Add a new medicine to the database."""
    drug = request.form.get('drug', '').strip()
    instructions = request.form.get('instructions', '').strip()
    
    if not drug or not instructions:
        flash('Both drug name and instructions are required', 'error')
        return redirect(url_for('admin'))
    
    if add_drug(drug, instructions):
        flash(f'Successfully added {drug}', 'success')
    else:
        flash(f'Failed to add {drug}. It may already exist.', 'error')
    
    return redirect(url_for('admin'))

@app.route('/admin/update', methods=['POST'])
def admin_update():
    """Update an existing medicine in the database."""
    drug_id = request.form.get('id')
    drug = request.form.get('drug', '').strip()
    instructions = request.form.get('instructions', '').strip()
    
    if not drug_id or not drug or not instructions:
        flash('All fields are required', 'error')
        return redirect(url_for('admin'))
    
    if update_drug(int(drug_id), drug, instructions):
        flash(f'Successfully updated {drug}', 'success')
    else:
        flash(f'Failed to update {drug}', 'error')
    
    return redirect(url_for('admin'))

@app.route('/admin/delete', methods=['POST'])
def admin_delete():
    """Delete a medicine from the database."""
    drug_id = request.form.get('id')
    
    if not drug_id:
        return jsonify({'success': False, 'error': 'No ID provided'})
    
    if delete_drug(int(drug_id)):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to delete'})

@app.route('/admin/import-docx', methods=['POST'])
def admin_import_docx():
    """Import medicines from a DOCX file."""
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('admin'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('admin'))
    
    if file and file.filename.endswith('.docx'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        count = load_from_docx_to_db(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return redirect(url_for('admin', message=f'Successfully imported {count} medicines'))
    else:
        flash('Invalid file type. Please upload a .docx file', 'error')
    
    return redirect(url_for('admin'))

@app.route('/api/database')
def api_database():
    """Return all medicines as JSON (for debugging)"""
    medicines = get_all_drugs_dict()
    return jsonify(medicines)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
