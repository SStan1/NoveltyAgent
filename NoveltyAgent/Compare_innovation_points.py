import os
import re
import openai
from ragflow_sdk import RAGFlow
import time
import traceback
import fitz

class _SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def format_prompt(template: str, **kwargs) -> str:
    return template.format_map(_SafeDict(**kwargs))

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text")
        doc.close()
        return full_text
    except Exception as e:
        print(f"[ERROR] Error extracting text from PDF {pdf_path}: {e}")
        return None

def generate_queries_from_innovation_point(client, innovation_point, config, paper_name, point_num):
    system_prompt = config['prompts']['query_generation']['system_prompt']
    user_prompt = config['prompts']['query_generation']['user_prompt'].format(
        paper_name=paper_name,
        point_num=point_num,
        innovation_point=innovation_point.strip()
    )

    max_retries = config['llm_config']['max_retries']
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries} to generate queries...")
            response = client.chat.completions.create(
                model=config['llm_config']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config['llm_config']['temperature']
            )
            response_content = response.choices[0].message.content
            lines = response_content.strip().split('\n')
            queries = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if re.match(r'^\d+\.\s*', line.strip())]

            if len(queries) >= 6:
                return queries[:6]
            elif len(queries) >= 3:
                while len(queries) < 6:
                    queries.append(innovation_point.strip())
                return queries
            else:
                continue
        except Exception as e:
            if attempt >= max_retries - 1:
                break
    return [innovation_point.strip()] * 6

def get_knowledge_from_ragflow_multiple_queries(rag_object, queries, dataset_name, config):
    all_chunks = []
    for i, query in enumerate(queries, 1):
        chunks = get_knowledge_from_ragflow(rag_object, query, dataset_name, config)
        if chunks:
            all_chunks.extend(chunks)
    unique_chunks = list(set(all_chunks))
    all_knowledge = ""
    for i, chunk in enumerate(unique_chunks, 1):
        all_knowledge += f"=== Chunk {i} ===\n{chunk}\n\n"
    return all_knowledge

def get_knowledge_from_ragflow(rag_object, query, dataset_name, config):
    try:
        datasets = rag_object.list_datasets(name=dataset_name)
        if not datasets:
            return []
        dataset = datasets[0]
        rag_config = config['rag']
        results = list(rag_object.retrieve(
            question=query,
            dataset_ids=[dataset.id],
            page=rag_config['page'],
            page_size=rag_config['page_size'],
            similarity_threshold=rag_config['similarity_threshold'],
            vector_similarity_weight=rag_config['vector_similarity_weight'],
            top_k=rag_config['top_k'],
            rerank_id=rag_config['rerank_id'],
            keyword=rag_config['keyword']
        ))
        chunks = []
        for c in results:
            try:
                doc = dataset.list_documents(id=c.document_id)
                document_name = doc[0].name if doc else "Unknown Document"
            except Exception:
                document_name = "Unknown Document"
            chunk_text = f"Source Document: {document_name}\nContent: {c.content}"
            chunks.append(chunk_text)
        return chunks
    except Exception as e:
        print(f"[ERROR] An error occurred in get_knowledge_from_ragflow: {e}")
        return []

def truncate_paper_text(paper_text, max_chars=50000):
    if len(paper_text) <= max_chars:
        return paper_text
    sections_to_keep = []
    abstract_match = re.search(r'(?i)(abstract\s*\n.*?)(?=\n\s*\d+\.?\s*introduction|\n\s*1\.?\s+)', paper_text, re.DOTALL)
    if abstract_match:
        sections_to_keep.append(abstract_match.group(1)[:5000])
    intro_match = re.search(r'(?i)(\d+\.?\s*introduction\s*\n.*?)(?=\n\s*\d+\.?\s*\w)', paper_text, re.DOTALL)
    if intro_match:
        sections_to_keep.append(intro_match.group(1)[:10000])
    method_match = re.search(r'(?i)(\d+\.?\s*(?:method|methodology|approach|proposed)\s*\n.*?)(?=\n\s*\d+\.?\s*\w)', paper_text, re.DOTALL)
    if method_match:
        sections_to_keep.append(method_match.group(1)[:15000])
    
    if sections_to_keep:
        truncated = "\n\n[...sections extracted...]\n\n".join(sections_to_keep)
        if len(truncated) < max_chars:
            remaining = max_chars - len(truncated)
            truncated = paper_text[:remaining//2] + "\n\n[...truncated...]\n\n" + truncated
        return truncated[:max_chars]
    else:
        return paper_text[:max_chars] + "\n\n[...content truncated due to length...]"

def parse_innovation_points(content):
    points = []
    matches = re.finditer(r'^\s*(\d+)\.\s*(.*?)(?=(?:\n\s*\d+\.\s)|$)', content, re.DOTALL | re.MULTILINE)
    for match in matches:
        points.append(match.group(2).strip())
    if not points and content:
        lines = content.strip().split('\n')
        current_point = ""
        for line in lines:
            if re.match(r'^\d+\.\s*', line):
                if current_point:
                    points.append(current_point.strip())
                current_point = line
            elif current_point:
                current_point += "\n" + line
        if current_point:
            points.append(current_point.strip())
    return [re.sub(r'^\d+\.\s*', '', p).strip() for p in points]

def limit_innovation_points(innovation_points, max_points=5):
    if len(innovation_points) <= max_points:
        return innovation_points
    return innovation_points[:max_points]

def compare_paper_innovations(config, paper_name, dataset_name, innovation_content, main_pdf_path=None):
    try:
        rag_object = RAGFlow(
            api_key=config['api']['api_key'],
            base_url=config['api']['base_url']
        )
        client = openai.OpenAI(
            api_key=config['api']['openai_api_key'],
            base_url=config['api']['openai_base_url'],
            timeout=config['api']['openai_timeout']
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize clients: {e}")
        return None

    print("[INFO] Starting innovation point comparison...")

    # Read full-text analysis mode toggle
    use_full_text = config.get('use_full_text_in_comparison', True)

    original_paper_text = ""
    if use_full_text:
        print("[INFO] Full text analysis mode: ON (using original paper text in comparison prompts)")
        if main_pdf_path and os.path.exists(main_pdf_path):
            original_paper_text = extract_text_from_pdf(main_pdf_path)
            if original_paper_text:
                original_paper_text = truncate_paper_text(original_paper_text, max_chars=50000)
                print(f"   Extracted and truncated original paper text: {len(original_paper_text)} characters")
            else:
                print("   [WARN] Failed to extract text from main PDF, proceeding without full text.")
        else:
            print("   [WARN] Main PDF path not available, proceeding without full text.")
    else:
        print("[INFO] Full text analysis mode: OFF (saving tokens, using only innovation descriptions and RAG knowledge)")

    innovation_points = parse_innovation_points(innovation_content)
    if not innovation_points:
        return None

    innovation_points = limit_innovation_points(innovation_points, max_points=5)
    comparison_data = []

    # Select prompt key based on mode
    if use_full_text:
        user_prompt_key = 'user_prompt'
    else:
        user_prompt_key = 'user_prompt_no_fulltext'

    for i, point in enumerate(innovation_points, 1):
        print(f"\n{'='*60}\n[INFO] Processing innovation point {i}/{len(innovation_points)}\n{'='*60}")
        try:
            queries = generate_queries_from_innovation_point(client, point, config, paper_name, i)
            knowledge = get_knowledge_from_ragflow_multiple_queries(rag_object, queries, dataset_name, config)

            if not knowledge:
                print(f"[WARN] No knowledge retrieved for point {i}.")
                continue

            system_prompt = format_prompt(
                config['prompts']['innovation_comparison']['system_prompt'],
                knowledge=knowledge
            )
            user_prompt = format_prompt(
                config['prompts']['innovation_comparison'][user_prompt_key],
                paper_name=paper_name,
                point_num=i,
                innovation_point=point.strip(),
                original_paper_text=original_paper_text
            )

            response = client.chat.completions.create(
                model=config['llm_config']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config['llm_config']['temperature']
            )

            full_response = response.choices[0].message.content
            comparison_data.append({'point_number': i, 'content': full_response})
            print(f"[OK] Comparison for point {i} completed.")
            time.sleep(1)
        except Exception as e:
            print(f"[ERROR] Error processing innovation point {i}: {e}")
            continue

    return comparison_data