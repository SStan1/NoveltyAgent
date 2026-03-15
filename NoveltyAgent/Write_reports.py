import os
import re
import time
from datetime import datetime
from openai import OpenAI

class InnovationReportGenerator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            api_key=config['api']['openai_api_key'],
            base_url=config['api']['openai_base_url'],
            timeout=config['api']['openai_timeout']
        )

    def read_innovation_classifications(self, innovation_content):
        classifications = {}
        point_summaries = {}
        if not innovation_content:
            return classifications, point_summaries

        pattern1 = r'^(\d+)\.\s*\(([^)]+)\)'
        pattern2 = r'(\d+)\.\s*\(([^)]+)\)\s*Innovation summary:'

        lines = innovation_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue
            match1 = re.match(pattern1, line)
            if match1:
                point_num = int(match1.group(1))
                classification = match1.group(2).strip()
                if point_num not in classifications:
                    classifications[point_num] = classification

        if not classifications:
            matches = re.finditer(pattern2, innovation_content)
            for match in matches:
                point_num = int(match.group(1))
                classification = match.group(2).strip()
                classifications[point_num] = classification

        if not classifications:
            matches = re.finditer(pattern1, innovation_content, re.MULTILINE)
            for match in matches:
                point_num = int(match.group(1))
                classification = match.group(2).strip()
                if point_num not in classifications:
                    classifications[point_num] = classification

        summary_block_pattern = re.compile(
            r'^\s*(\d+)\.\s*(.*?)(?=(?:\n\s*\d+\.\s)|$)',
            re.DOTALL | re.MULTILINE
        )
        summary_pattern = re.compile(
            r'Innovation summary:\s*(.*?)(?=\n\s*\n|\n\s*[A-Z][A-Za-z ]{0,30}:|$)',
            re.IGNORECASE | re.DOTALL
        )

        for block in summary_block_pattern.finditer(innovation_content):
            point_num = int(block.group(1))
            block_body = block.group(2).strip()
            summary_match = summary_pattern.search(block_body)
            if summary_match:
                summary_text = summary_match.group(1).strip()
            else:
                summary_text = block_body
            summary_text = re.sub(r'^\(([^)]+)\)\s*', '', summary_text).strip()
            point_summaries[point_num] = summary_text

        return classifications, point_summaries

    def detect_format_type(self, content):
        if re.search(r'\*\*[a-d]\)\s+', content): return 'new_bold'
        elif re.search(r'####\s+[a-d]\)', content): return 'old'
        elif re.search(r'###\s+\d+\.\s+Point-wise Novelty Analysis', content): return 'new'
        elif re.search(r'a\)\s*(?:Restatement|Claimed Novelty):', content, re.I): return 'plain'
        return 'new_bold'

    def extract_comparison_sections(self, content):
        fmt = self.detect_format_type(content)
        sections = {}
        if fmt == 'new_bold':
            a = re.search(r'\*\*a\)\s*(?:Claimed Novelty|Restatement):?\*\*\s*(.*?)(?=\*\*b\)|$)', content, re.S | re.I)
            b = re.search(r'\*\*b\)\s*Similarities.*?:?\*\*\s*(.*?)(?=\*\*c\)|$)', content, re.S | re.I)
            c = re.search(r'\*\*c\)\s*Unique Differences:?\*\*\s*(.*?)(?=\*\*d\)|$)', content, re.S | re.I)
            d = re.search(r'\*\*d\)\s*Details of Unique Differences:?\*\*\s*(.*?)$', content, re.S | re.I)
        elif fmt == 'old':
            a = re.search(r'####\s+a\)\s*(?:Restatement|Claimed Novelty):?\s*(.*?)(?=\n####\s+[b-d]\)|$)', content, re.S | re.I)
            b = re.search(r'####\s+b\)\s*Similarities:?\s*(.*?)(?=\n####\s+[c-d]\)|$)', content, re.S | re.I)
            c = re.search(r'####\s+c\)\s*Unique Differences:?\s*(.*?)(?=\n####\s+d\)|$)', content, re.S | re.I)
            d = re.search(r'####\s+d\)\s*Details of Unique Differences:?\s*(.*?)$', content, re.S | re.I)
        elif fmt == 'new':
            comp = re.search(r'###\s+\d+\.\s+Point-wise Novelty Analysis\s*(.*?)(?=\n###\s+\d+\.|$)', content, re.S | re.I)
            data = comp.group(1) if comp else content
            a = re.search(r'####\s+a\)\s*(?:Claimed Novelty|Restatement):?\s*(.*?)(?=\n####\s+[b-d]\)|$)', data, re.S | re.I)
            b = re.search(r'####\s+b\)\s*Similarities:?\s*(.*?)(?=\n####\s+[c-d]\)|$)', data, re.S | re.I)
            c = re.search(r'####\s+c\)\s*Unique Differences:?\s*(.*?)(?=\n####\s+d\)|$)', data, re.S | re.I)
            d = re.search(r'####\s+d\)\s*Details of Unique Differences:?\s*(.*?)$', data, re.S | re.I)
        else:
            a = re.search(r'a\)\s*(?:Restatement|Claimed Novelty):?\s*(.*?)(?=\n\s*b\)|$)', content, re.S | re.I)
            b = re.search(r'b\)\s*Similarities:?\s*(.*?)(?=\n\s*c\)|$)', content, re.S | re.I)
            c = re.search(r'c\)\s*Unique Differences:?\s*(.*?)(?=\n\s*d\)|$)', content, re.S | re.I)
            d = re.search(r'd\)\s*Details of Unique Differences:?\s*(.*?)$', content, re.S | re.I)
            
        sections['a'] = a.group(1).strip() if a else ''
        sections['b'] = b.group(1).strip() if b else ''
        sections['c'] = c.group(1).strip() if c else ''
        sections['d'] = d.group(1).strip() if d else ''
        return sections

    def generate_draft_report(self, paper_name, summary_text, comparison_data, classifications, point_summaries):
        draft_parts = []
        draft_parts.append("## 1. Paper Content Summary\n\n")
        draft_parts.append(f"{summary_text}\n\n")
        draft_parts.append("## 2. Point-wise Novelty Analysis\n\n")

        sorted_data = sorted(comparison_data, key=lambda x: x.get('point_number', 0))
        section_idx = 1

        for data in sorted_data:
            n = data.get('point_number', 'N/A')
            cls = classifications.get(n, "Unknown")
            sec = self.extract_comparison_sections(data['content'])
            draft_parts.append(f"### 2.{section_idx}. Novelty Point {n}: (Classification: {cls})\n\n")
            if sec.get('a'):
                draft_parts.append(f"**a) Claimed Novelty:**\nClassification: {cls}.\n\nExplanation: {sec['a']}\n\n")
            if sec.get('b'):
                draft_parts.append(f"**b) Similarities:**\n{sec['b']}\n\n")
            if sec.get('c'):
                draft_parts.append(f"**c) Unique Differences:**\n{sec['c']}\n\n")
            if sec.get('d'):
                draft_parts.append(f"**d) Details of Unique Differences:**\n{sec['d']}\n\n")
            section_idx += 1

        return "".join(draft_parts)

    def generate_section3(self, draft_section2):
        prompt = self.config['prompts']['report_section3'].replace('{draft_section2}', draft_section2)
        retries = self.config['llm_config']['max_retries']
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config['llm_config']['model'],
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content.strip()
                if content: return content
            except Exception as e:
                print(f"[WARN] Section 3 generation attempt {attempt+1} failed: {e}")
                if attempt < retries - 1: time.sleep(self.config['llm_config']['retry_delay'])
        return "Section 3 generation failed."

    def format_references_locally(self, draft_report):
        patterns = [
            r'##document_name:\s*[^$]+?\$\$',
            r'##REF_[^$]+?\$\$',
            r'##[A-Za-z0-9_\-\s]+\.(?:pdf|txt|docx?|md)(?:##|\$\$)'
        ]
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        full_pattern = re.compile(combined_pattern, re.MULTILINE)

        all_refs = []
        for match in full_pattern.finditer(draft_report):
            ref_marker = match.group(0)
            if ref_marker.startswith('##'):
                all_refs.append(ref_marker)

        unique_refs = []
        seen = set()
        for ref in all_refs:
            ref_normalized = ' '.join(ref.split())
            if ref_normalized not in seen:
                seen.add(ref_normalized)
                unique_refs.append(ref)

        print(f"[INFO] Found {len(unique_refs)} unique references")

        ref_mapping = {ref: f"[{i}]" for i, ref in enumerate(unique_refs, 1)}
        sorted_refs = sorted(ref_mapping.items(), key=lambda x: len(x[0]), reverse=True)

        formatted_report = draft_report
        for ref_marker, ref_number in sorted_refs:
            if ref_marker in formatted_report:
                formatted_report = formatted_report.replace(ref_marker, ref_number)

        if unique_refs:
            references_section = "\n\n---\n\n## References\n\n"
            for i, ref_marker in enumerate(unique_refs, 1):
                if ref_marker.startswith('##document_name:'):
                    clean_ref = ref_marker.replace('##document_name:', '').replace('$$', '').strip()
                elif ref_marker.startswith('##REF_'):
                    clean_ref = ref_marker.replace('##', '').replace('$$', '').strip()
                else:
                    clean_ref = ref_marker.replace('##', '').strip()
                    if clean_ref.endswith('$$'): clean_ref = clean_ref[:-2].strip()
                references_section += f"[{i}] {clean_ref}\n"
            references_section += "\n---"
        else:
            references_section = "\n\n---\n\n## References\n\nNo external references cited in this analysis.\n\n---"

        return formatted_report + references_section

    def generate_comprehensive_report(self, paper_name, summary_text, innovation_content, comparison_data):
        classifications, point_summaries = self.read_innovation_classifications(innovation_content)
        
        print(f"[INFO] Step 1: Generating draft (Sections 1-2)")
        draft = self.generate_draft_report(paper_name, summary_text, comparison_data, classifications, point_summaries)
        
        sec2_match = re.search(r'## 2\. Point-wise Novelty Analysis\n\n(.*)', draft, re.S)
        sec2_text = sec2_match.group(1).strip() if sec2_match else ""
        
        print("[INFO] Step 2: Generating Section 3")
        sec3 = self.generate_section3(sec2_text)
        full_draft = draft + "\n" + sec3
        
        print("[INFO] Step 3: Formatting references")
        final = self.format_references_locally(full_draft)
        
        point_count = len(comparison_data)

        # Only return the report body; header/footer is handled by the polisher or save step
        return final, point_count