import re
import time
from datetime import datetime
from openai import OpenAI

class ReportPolisher:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            api_key=config['api']['openai_api_key'],
            base_url=config['api']['openai_base_url'],
            timeout=config['api'].get('openai_timeout', 900)
        )
        self.model = config['llm_config']['model']
        self.temperature = config['llm_config']['temperature']
        self.max_retries = config['llm_config']['max_retries']
        self.retry_delay = config['llm_config']['retry_delay']
        # Exponential backoff parameters
        self.retry_delay_multiplier = 1.5
        self.max_retry_delay = 60

    def _calculate_retry_delay(self, attempt):
        """Calculate retry delay with exponential backoff."""
        delay = self.retry_delay * (self.retry_delay_multiplier ** attempt)
        return min(delay, self.max_retry_delay)

    def validate_polished_report(self, original_content, polished_content):
        """
        Validate that the polished report retains required structures and core content.
        """
        warnings = []
        
        # 1. Check required report structures
        required_report_structures = [
            'a) Claimed Novelty:',
            'b) Similarities:',
            'c) Unique Differences:',
            'd) Details of Unique Differences:',
        ]
        
        for structure in required_report_structures:
            if structure not in polished_content:
                warnings.append(f"Missing required report structure: '{structure}'")
        
        # 2. Check length ratio (polished should not be too short)
        if len(original_content) > 0:
            length_ratio = len(polished_content) / len(original_content)
            if length_ratio < 0.6:
                warnings.append(f"Polished report is too short ({length_ratio:.1%} of original)")
        
        # 3. Check reference count
        original_refs = re.findall(r'\[\d+\]', original_content)
        polished_refs = re.findall(r'\[\d+\]', polished_content)
        
        if len(original_refs) > 0 and len(polished_refs) < len(original_refs) * 0.9:
            warnings.append(f"Reference count dropped significantly: Original has {len(original_refs)}, Polished has {len(polished_refs)}")
        
        # 4. Check major section titles
        required_sections = ['Paper Content Summary', 'Point-wise Novelty Analysis', 'Novelty Summary']
        for section in required_sections:
            if section in original_content and section not in polished_content:
                warnings.append(f"Required section missing: {section}")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings

    def clean_polished_content(self, polished_content):
        cleaned = polished_content
        preamble_patterns = [
            r'^Here is the polished report[:\s]*\n*',
            r'^I have polished the report[:\s]*\n*',
            r'^Below is the polished[:\s]*\n*',
            r'^The polished report[:\s]*\n*',
        ]
        for pattern in preamble_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        ending_patterns = [
            r'\n*---\s*End of [Pp]olished [Rr]eport\s*---\s*$',
            r'\n*I hope this helps[.!]*\s*$',
            r'\n*Let me know if you need[^.]*[.]\s*$',
            r'\n*Please let me know[^.]*[.]\s*$',
        ]
        for pattern in ending_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        if not cleaned.strip().startswith('## 1.'):
            match = re.search(r'(## 1\. Paper Content Summary)', cleaned)
            if match: cleaned = cleaned[match.start():]
        return cleaned.strip()

    def polish_single_report(self, report_content, paper_name, num_points):
        print(f"\n{'='*100}\nPOLISHING REPORT\n{'='*100}\n")
        print(f"[INFO] Model: {self.model}")
        print(f"[INFO] Report length: {len(report_content)} characters")
        
        prompt = self.config['prompts']['polish']['system_prompt'].replace('{report_content}', report_content)
        
        for attempt in range(self.max_retries):
            try:
                print(f"\n[INFO] Attempt {attempt + 1}/{self.max_retries}")
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                
                elapsed_time = time.time() - start_time
                print(f"[INFO] API response received in {elapsed_time:.1f} seconds")
                
                polished_content = response.choices[0].message.content.strip()
                polished_content = self.clean_polished_content(polished_content)
                
                is_valid, warnings = self.validate_polished_report(report_content, polished_content)
                
                if warnings:
                    print(f"[WARN] Validation warnings detected:")
                    for warning in warnings:
                        print(f"       - {warning}")
                
                if not is_valid and attempt < self.max_retries - 1:
                    # Use exponential backoff
                    retry_delay = self._calculate_retry_delay(attempt)
                    print(f"[WARN] Validation failed, will retry after {retry_delay:.0f} seconds...")
                    time.sleep(retry_delay)
                    continue
                
                if is_valid:
                    print(f"[SUCCESS] Report polished successfully")
                else:
                    print(f"[WARN] Report polished with warnings (used anyway after {self.max_retries} attempts)")
                
                print(f"[INFO] Original length: {len(report_content)} characters")
                print(f"[INFO] Polished length: {len(polished_content)} characters")
                print(f"[INFO] Length ratio: {len(polished_content)/max(len(report_content),1):.1%}")
                
                # Add header/footer (final output format)
                final_output = "=" * 100 + "\n"
                final_output += "POLISHED COMPREHENSIVE NOVELTY SYNTHESIS REPORT\n"
                final_output += "=" * 100 + "\n"
                final_output += f"Paper: {paper_name}\n"
                final_output += f"Number of Novelty Points Analyzed: {num_points}\n"
                final_output += "=" * 100 + "\n\n"
                final_output += polished_content
                final_output += "\n\n" + "=" * 100 + "\n"
                final_output += "End of Polished Report\n"
                final_output += "=" * 100 + "\n"
                
                return final_output
                
            except Exception as e:
                elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                print(f"[ERROR] Attempt {attempt + 1} failed after {elapsed_time:.1f}s: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff with error-type-specific wait strategies
                    retry_delay = self._calculate_retry_delay(attempt)
                    
                    error_str = str(e).lower()
                    if 'rate limit' in error_str or 'too many requests' in error_str:
                        retry_delay = max(retry_delay, 60)
                        print(f"[WARN] Rate limit detected, extending wait time")
                    elif 'timeout' in error_str or 'timed out' in error_str:
                        retry_delay = max(retry_delay, 30)
                        print(f"[WARN] Timeout detected, will retry with extended wait")
                    elif 'overloaded' in error_str or 'capacity' in error_str:
                        retry_delay = max(retry_delay, 90)
                        print(f"[WARN] Server overloaded, extending wait time significantly")
                    
                    print(f"[INFO] Waiting {retry_delay:.0f} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    print(f"[ERROR] All retries exhausted. Returning original content with header.")
        
        # All retries failed, return original content with header/footer
        print("[ERROR] Failed to polish report, returning original with header.")
        final_output = "=" * 100 + "\n"
        final_output += "COMPREHENSIVE NOVELTY SYNTHESIS REPORT (UNPOLISHED)\n"
        final_output += "=" * 100 + "\n"
        final_output += f"Paper: {paper_name}\n"
        final_output += f"Number of Novelty Points Analyzed: {num_points}\n"
        final_output += "=" * 100 + "\n\n"
        final_output += report_content
        final_output += "\n\n" + "=" * 100 + "\n"
        final_output += "End of Report\n"
        final_output += "=" * 100 + "\n"
        return final_output