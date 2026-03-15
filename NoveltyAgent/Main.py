import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Use the same clean_filename as the crawling/database pipeline
# to ensure naming consistency across all stages
from clawer_papers import clean_filename


class Logger:
    def __init__(self, filename="pipeline.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# [REMOVED] sanitize_filename — replaced by clean_filename from clawer_papers
# to unify the cleaning logic with crawling and database construction.

def find_main_pdf_path(paper_folder_path):
    for filename in os.listdir(paper_folder_path):
        if filename.lower().startswith("main_") and filename.lower().endswith(".pdf"):
            return os.path.join(paper_folder_path, filename)
    return None

def derive_report_base_name(main_pdf_path):
    """
    Derive the report base name from the MAIN PDF filename.

    MAIN PDF naming (set by clawer_papers):
        MAIN_<clean_filename(title)[:100]>.pdf

    Dataset naming (set by Create_database_and_parse.upload_pdfs_to_ragflow):
        main_pdf.stem[4:24] + "_main_only_dataset"  /  "_dataset"
        i.e.  "_" + clean_filename(title)[:19] + suffix

    Evaluation.infer_dataset_name reconstructs the dataset name as:
        "_" + paper_name[:19] + "_main_only_dataset"  /  "_dataset"
        where paper_name = report filename without extension

    Therefore, report filename (without extension) must equal
    clean_filename(title)[:100], which is exactly MAIN PDF stem
    with the "MAIN_" prefix stripped.
    """
    main_pdf_stem = os.path.splitext(os.path.basename(main_pdf_path))[0]
    if main_pdf_stem.startswith("MAIN_"):
        return main_pdf_stem[5:]
    return main_pdf_stem

# ================= Checkpoint utilities for resume support =================

def get_checkpoint_dir(base_result_dir):
    """Create a checkpoints subdirectory inside the result dir for saving intermediate outputs."""
    checkpoint_dir = os.path.join(base_result_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(checkpoint_dir, filename, content):
    """Save intermediate results to the checkpoint directory."""
    filepath = os.path.join(checkpoint_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(content))
    print(f"  [INFO] Checkpoint saved: {filepath}")

def load_checkpoint(checkpoint_dir, filename, as_json=False):
    """Load intermediate results from the checkpoint directory; returns None if missing or empty."""
    filepath = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if as_json:
                data = json.load(f)
                if not data:
                    return None
                return data
            else:
                content = f.read()
                if content and content.strip():
                    return content
                return None
    except Exception as e:
        print(f"  [WARN] Failed to load checkpoint {filepath}: {e}")
        return None

# ==================================================================

def main():
    config = load_config()
    paper_name = config.get('paper_name')
    if not paper_name:
        print("[ERROR] 'paper_name' is required in config.json")
        return

    # ================= Set up result directory =================
    result_dir = config['paths'].get('result_dir', './result')
    os.makedirs(result_dir, exist_ok=True)

    # [CHANGED] Use clean_filename (same as crawling pipeline) instead of sanitize_filename
    safe_paper_name = clean_filename(paper_name)

    # Create a dedicated result folder for this paper
    paper_result_dir = os.path.join(result_dir, safe_paper_name)
    os.makedirs(paper_result_dir, exist_ok=True)

    # Create checkpoints directory (under result/paper_name/checkpoints)
    checkpoint_dir = get_checkpoint_dir(paper_result_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(paper_result_dir, f"{safe_paper_name}_{timestamp}.log")

    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    # ================================================

    print(f"\n{'='*100}")
    print(f"STARTING PIPELINE FOR PAPER: {paper_name}")
    print(f"PAPER RESULT DIR: {paper_result_dir}")
    print(f"LOG FILE: {log_file_path}")
    print(f"CHECKPOINT DIR: {checkpoint_dir}")
    print(f"{'='*100}\n")

    # 1. Crawl papers
    print("\n[STEP 1/7] Checking/Downloading Paper...")
    from clawer_papers import download_paper_if_needed
    paper_folder_path = download_paper_if_needed(config)
    if not paper_folder_path:
        print("[ERROR] Failed to get or download paper.")
        return

    # Get main_pdf_path early since multiple steps need it
    main_pdf_path = find_main_pdf_path(paper_folder_path)
    if not main_pdf_path:
        print(f"[ERROR] MAIN_*.pdf not found in {paper_folder_path}")
        return

    # [NEW] Derive report base name from MAIN PDF so that
    # Evaluation.infer_dataset_name can correctly reconstruct the RAGFlow dataset names.
    # This is the single source of truth for the report filename.
    report_base_name = derive_report_base_name(main_pdf_path)
    print(f"[INFO] Report base name (from MAIN PDF): {report_base_name}")

    # 2. Upload and parse
    print("\n[STEP 2/7] Uploading and Parsing PDFs in RAGFlow...")
    from Create_database_and_parse import upload_pdfs_to_ragflow, wait_for_parsing_completion
    main_dataset, all_dataset = upload_pdfs_to_ragflow(config, paper_folder_path)
    if not all_dataset:
        print("[ERROR] Failed to create dataset.")
        return

    all_parsing_success = wait_for_parsing_completion(all_dataset, config)
    if not all_parsing_success:
        print(f"[ERROR] Dataset parsing failed for all_dataset. Aborting pipeline.")
        return

    if main_dataset:
        main_parsing_success = wait_for_parsing_completion(main_dataset, config)
        if not main_parsing_success:
            print(f"[WARN] Main-only dataset parsing failed, continuing with all_dataset only.")

    # 3. Extract Summary and Innovation (with checkpoint/resume support)
    print("\n[STEP 3/7] Extracting Summary and Innovation Points...")

    summary_result = load_checkpoint(checkpoint_dir, "summary.txt")
    innovation_result = load_checkpoint(checkpoint_dir, "innovation_points.txt")

    if summary_result:
        print("  [OK] Loaded summary from checkpoint, skipping summary extraction.")
    else:
        from Generate_Mainpaper_summary import get_paper_summary
        summary_result = get_paper_summary(config, paper_name, main_pdf_path)
        if summary_result:
            save_checkpoint(checkpoint_dir, "summary.txt", summary_result)

    if innovation_result:
        print("  [OK] Loaded innovation points from checkpoint, skipping innovation extraction.")
    else:
        from Generate_innovation_points import get_paper_innovation
        innovation_result = get_paper_innovation(config, paper_name, main_pdf_path)
        if innovation_result:
            save_checkpoint(checkpoint_dir, "innovation_points.txt", innovation_result)

    if not summary_result or not innovation_result:
        print("[ERROR] Failed to extract summary or innovation points.")
        return

    # 4. Compare Innovation (with checkpoint/resume support)
    print("\n[STEP 4/7] Comparing Innovation Points via RAG...")

    comparison_data = load_checkpoint(checkpoint_dir, "comparison_data.json", as_json=True)

    if comparison_data:
        print("  [OK] Loaded comparison data from checkpoint, skipping comparison.")
    else:
        from Compare_innovation_points import compare_paper_innovations
        comparison_data = compare_paper_innovations(
            config, paper_name, all_dataset.name, innovation_result, main_pdf_path
        )
        if comparison_data:
            save_checkpoint(checkpoint_dir, "comparison_data.json", comparison_data)

    if not comparison_data:
        print("[ERROR] Failed to compare innovation points.")
        return

    # 5. Generate report (with checkpoint/resume support)
    print("\n[STEP 5/7] Generating Comprehensive Report...")

    initial_report = load_checkpoint(checkpoint_dir, "initial_report.txt")
    point_count = len(comparison_data)

    if initial_report:
        print("  [OK] Loaded initial report from checkpoint, skipping report generation.")
    else:
        from Write_reports import InnovationReportGenerator
        generator = InnovationReportGenerator(config)
        report_content, point_count = generator.generate_comprehensive_report(
            paper_name, summary_result, innovation_result, comparison_data
        )

        if report_content:
            initial_report = report_content
            save_checkpoint(checkpoint_dir, "initial_report.txt", initial_report)

    if not initial_report:
        print("[ERROR] Failed to generate report.")
        return

    # 6. Validate citations (with checkpoint/resume support)
    print("\n[STEP 6/7] Validating and Correcting Citations...")

    validated_report = load_checkpoint(checkpoint_dir, "validated_report.txt")

    if validated_report:
        print("  [OK] Loaded validated report from checkpoint, skipping citation validation.")
    else:
        from Validate_and_correct_citations import CitationValidator
        validator = CitationValidator(config)
        validated_report = validator.validate_and_correct_single_report(initial_report, paper_folder_path)
        if validated_report:
            save_checkpoint(checkpoint_dir, "validated_report.txt", validated_report)

    if not validated_report:
        print("[WARN] Citation validation returned empty, using initial report.")
        validated_report = initial_report

    # 7. Polish report
    print("\n[STEP 7/7] Polishing Final Report...")
    from Final_polish import ReportPolisher
    polisher = ReportPolisher(config)
    polished_report = polisher.polish_single_report(validated_report, paper_name, point_count)

    # 8. Save final report to dedicated result directory

    # so that Evaluation.infer_dataset_name(paper_name) works correctly:
    #   paper_name = os.path.splitext("report_base_name.txt")[0] = report_base_name
    #   paper_name[:19] == clean_filename(title)[:19] == main_pdf.stem[5:24]
    #   → dataset name match guaranteed
    final_path = os.path.join(paper_result_dir, f"{report_base_name}.txt")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(polished_report)

    print(f"\n{'='*100}")
    print(f"[OK] PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Paper Result Directory: {paper_result_dir}")
    print(f"Final Report Saved To: {final_path}")
    print(f"Run Log Saved To: {log_file_path}")
    print(f"Checkpoints Saved In: {checkpoint_dir}")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()