#!/usr/bin/env python3
"""
Generate Word (.docx) lab record files for all 9 NLP categories.
Reads the Markdown lab records and produces styled Word documents.
"""

import os
import re
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
MD_DIR   = os.path.join(ROOT, 'docs', 'lab_records')
DOCX_DIR = os.path.join(ROOT, 'docs', 'lab_records')

# ── Color Palette ──────────────────────────────────────────────────────────────
COLOR_HEADING1  = RGBColor(0x1F, 0x49, 0x7D)   # Dark navy
COLOR_HEADING2  = RGBColor(0x2E, 0x74, 0xB5)   # Medium blue
COLOR_HEADING3  = RGBColor(0x5A, 0x96, 0xD4)   # Light blue
COLOR_CODE_BG   = RGBColor(0xF2, 0xF2, 0xF2)   # Light grey
COLOR_TABLE_HDR = RGBColor(0x2E, 0x74, 0xB5)   # Blue header
COLOR_ACCENT    = RGBColor(0xE5, 0x45, 0x45)   # Red accent
COLOR_BODY      = RGBColor(0x1A, 0x1A, 0x1A)   # Near black

FONT_BODY    = 'Calibri'
FONT_HEADING = 'Calibri Light'
FONT_CODE    = 'Courier New'


# ── Document Style Helpers ─────────────────────────────────────────────────────

def set_cell_background(cell, hex_color: str):
    """Set table cell background color."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)


def add_horizontal_rule(doc: Document, color: str = '2E74B5'):
    """Add a thin colored horizontal rule."""
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after  = Pt(6)
    pPr  = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'),   'single')
    bottom.set(qn('w:sz'),    '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), color)
    pBdr.append(bottom)
    pPr.append(pBdr)
    return para


def style_document(doc: Document):
    """Apply global document styles."""
    style = doc.styles['Normal']
    font  = style.font
    font.name = FONT_BODY
    font.size = Pt(11)
    font.color.rgb = COLOR_BODY

    # Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)


def add_cover_header(doc: Document, cat_num: int, title: str, subtitle: str = ''):
    """Add styled document header with category info."""
    # Top banner paragraph
    banner = doc.add_paragraph()
    banner.alignment = WD_ALIGN_PARAGRAPH.CENTER
    banner.paragraph_format.space_before = Pt(0)
    banner.paragraph_format.space_after  = Pt(4)
    run = banner.add_run('NLP Lab Record')
    run.font.name  = FONT_HEADING
    run.font.size  = Pt(11)
    run.font.color.rgb = RGBColor(0x70, 0x70, 0x70)
    run.font.italic = True

    # Category badge
    cat_para = doc.add_paragraph()
    cat_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cat_para.paragraph_format.space_before = Pt(0)
    cat_para.paragraph_format.space_after  = Pt(2)
    cat_run = cat_para.add_run(f'CATEGORY {cat_num}')
    cat_run.font.name  = FONT_HEADING
    cat_run.font.size  = Pt(14)
    cat_run.font.bold  = True
    cat_run.font.color.rgb = COLOR_HEADING2

    # Main title
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_para.paragraph_format.space_before = Pt(0)
    title_para.paragraph_format.space_after  = Pt(8)
    title_run = title_para.add_run(title)
    title_run.font.name  = FONT_HEADING
    title_run.font.size  = Pt(22)
    title_run.font.bold  = True
    title_run.font.color.rgb = COLOR_HEADING1

    if subtitle:
        sub_para = doc.add_paragraph()
        sub_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub_para.paragraph_format.space_after = Pt(4)
        sub_run = sub_para.add_run(subtitle)
        sub_run.font.name   = FONT_BODY
        sub_run.font.size   = Pt(11)
        sub_run.font.italic = True
        sub_run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    add_horizontal_rule(doc, '1F497D')
    doc.add_paragraph()  # spacer


def add_section_heading(doc: Document, text: str, level: int = 1) -> None:
    """Add a styled section heading."""
    para = doc.add_paragraph()
    if level == 1:
        para.paragraph_format.space_before = Pt(14)
        para.paragraph_format.space_after  = Pt(4)
        run = para.add_run(text.upper())
        run.font.name  = FONT_HEADING
        run.font.size  = Pt(14)
        run.font.bold  = True
        run.font.color.rgb = COLOR_HEADING1
        add_horizontal_rule(doc, '2E74B5')
    elif level == 2:
        para.paragraph_format.space_before = Pt(10)
        para.paragraph_format.space_after  = Pt(2)
        run = para.add_run(text)
        run.font.name  = FONT_HEADING
        run.font.size  = Pt(12)
        run.font.bold  = True
        run.font.color.rgb = COLOR_HEADING2
    elif level == 3:
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after  = Pt(2)
        run = para.add_run(text)
        run.font.name   = FONT_BODY
        run.font.size   = Pt(11)
        run.font.bold   = True
        run.font.italic = True
        run.font.color.rgb = COLOR_HEADING3


def add_body_paragraph(doc: Document, text: str) -> None:
    """Add a body paragraph with inline formatting support."""
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(1)
    para.paragraph_format.space_after  = Pt(4)
    _apply_inline(para, text)


def add_bullet(doc: Document, text: str, level: int = 0) -> None:
    """Add a bullet point."""
    para = doc.add_paragraph(style='List Bullet')
    para.paragraph_format.space_before = Pt(1)
    para.paragraph_format.space_after  = Pt(1)
    para.paragraph_format.left_indent  = Inches(0.25 * (level + 1))
    _apply_inline(para, text)


def add_code_block(doc: Document, code: str) -> None:
    """Add a styled code block."""
    lines = code.strip().split('\n')
    for line in lines:
        para = doc.add_paragraph()
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after  = Pt(0)
        para.paragraph_format.left_indent  = Inches(0.4)
        run = para.add_run(line if line else ' ')
        run.font.name = FONT_CODE
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0x1E, 0x1E, 0x2E)
        # Light grey background via shading
        pPr = para._p.get_or_add_pPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'),   'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'),  'F0F0F0')
        pPr.append(shd)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_table_from_md(doc: Document, md_table: str) -> None:
    """Parse and render a Markdown pipe table."""
    lines = [l.strip() for l in md_table.strip().split('\n') if l.strip()]
    # Filter separator row
    data_rows = [l for l in lines if not re.match(r'^\|[-| :]+\|$', l)]
    if not data_rows:
        return

    rows = []
    for line in data_rows:
        cells = [c.strip() for c in line.strip('|').split('|')]
        rows.append(cells)

    if not rows:
        return

    ncols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx]
        for c_idx, cell_text in enumerate(row_data):
            if c_idx >= ncols:
                break
            cell = row.cells[c_idx]
            cell.text = ''
            para = cell.paragraphs[0]
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.space_after  = Pt(2)
            # Strip inline bold from headers
            clean_text = re.sub(r'\*\*(.+?)\*\*', r'\1', cell_text)
            run = para.add_run(clean_text)
            run.font.name = FONT_BODY
            run.font.size = Pt(10)

            if r_idx == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_background(cell, '2E74B5')
            elif r_idx % 2 == 0:
                set_cell_background(cell, 'EBF3FB')

    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def _apply_inline(para, text: str):
    """Apply bold/italic/code inline markdown formatting to a paragraph."""
    # Split on inline code `...`, **bold**, *italic*
    pattern = re.compile(r'(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)')
    parts = pattern.split(text)
    for part in parts:
        if not part:
            continue
        run = para.add_run()
        run.font.name = FONT_BODY
        run.font.size = Pt(11)
        run.font.color.rgb = COLOR_BODY
        if part.startswith('`') and part.endswith('`'):
            run.text = part[1:-1]
            run.font.name  = FONT_CODE
            run.font.size  = Pt(10)
            run.font.color.rgb = RGBColor(0xC7, 0x25, 0x4F)
        elif part.startswith('**') and part.endswith('**'):
            run.text = part[2:-2]
            run.font.bold = True
        elif part.startswith('*') and part.endswith('*'):
            run.text = part[1:-1]
            run.font.italic = True
        else:
            run.text = part


# ── Markdown Parser → DOCX ────────────────────────────────────────────────────

def md_to_docx(md_path: str, doc: Document):
    """Parse a Markdown file and render it into a Document."""
    with open(md_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Heading 1 ──
        if line.startswith('# ') and not line.startswith('## '):
            # Skip the first H1 (used as document title already)
            i += 1
            continue

        # ── Heading 2 ──
        elif line.startswith('## '):
            add_section_heading(doc, line[3:].strip(), level=1)
            i += 1

        # ── Heading 3 ──
        elif line.startswith('### '):
            add_section_heading(doc, line[4:].strip(), level=2)
            i += 1

        # ── Heading 4 ──
        elif line.startswith('#### '):
            add_section_heading(doc, line[5:].strip(), level=3)
            i += 1

        # ── Code block ──
        elif line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            add_code_block(doc, '\n'.join(code_lines))

        # ── Table ──
        elif line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].startswith('|'):
                table_lines.append(lines[i])
                i += 1
            add_table_from_md(doc, '\n'.join(table_lines))

        # ── Bullet (- or * at start) ──
        elif re.match(r'^[-*] ', line):
            text = line[2:].strip()
            add_bullet(doc, text, level=0)
            i += 1

        # ── Numbered list ──
        elif re.match(r'^\d+\. ', line):
            text = re.sub(r'^\d+\. ', '', line).strip()
            add_bullet(doc, text, level=0)
            i += 1

        # ── Blank line ──
        elif line.strip() == '':
            i += 1

        # ── Normal paragraph ──
        else:
            text = line.strip()
            if text:
                add_body_paragraph(doc, text)
            i += 1


# ── Per-Category Metadata ──────────────────────────────────────────────────────

CATEGORY_META = {
    1: {
        'title': 'Text Preprocessing',
        'subtitle': 'Normalization · Tokenization · Stemming · Lemmatization',
        'datasets': 'Brown Corpus, Reuters Corpus (NLTK)',
    },
    2: {
        'title': 'Entropy, Cross-Entropy & Perplexity',
        'subtitle': 'Shannon Entropy · KL Divergence · Language Model Evaluation',
        'datasets': 'Brown Corpus, Reuters Corpus, 20 Newsgroups',
    },
    3: {
        'title': 'Rule-Based Morphological Analyzer',
        'subtitle': 'Prefix/Suffix Rules · POS Transformations',
        'datasets': 'Brown Corpus (NLTK)',
    },
    4: {
        'title': 'FSA-Based Morphological Analyzer',
        'subtitle': 'Finite State Automata · State Transitions · Inflection Generation',
        'datasets': 'Brown Corpus (NLTK)',
    },
    5: {
        'title': 'Unigram Language Model',
        'subtitle': 'MLE · Laplace Smoothing · Zipf\'s Law · Perplexity',
        'datasets': 'Brown Corpus, Reuters Corpus (NLTK)',
    },
    6: {
        'title': 'Bigram Language Model',
        'subtitle': 'Laplace · Good-Turing · Kneser-Ney Smoothing',
        'datasets': 'Brown Corpus, Reuters Corpus (NLTK)',
    },
    7: {
        'title': 'Neural Language Models',
        'subtitle': 'Spelling Correction · CBOW · Skip-gram Word2Vec',
        'datasets': 'Brown Corpus, Reuters Corpus (NLTK)',
    },
    8: {
        'title': 'Vector Semantics',
        'subtitle': 'TF-IDF · Cosine Similarity · PMI / PPMI',
        'datasets': 'Reuters Corpus (NLTK), Brown Corpus',
    },
    9: {
        'title': 'Word Embeddings & Visualization',
        'subtitle': 'Word2Vec · PCA · t-SNE · UMAP',
        'datasets': 'Brown Corpus + Reuters Corpus (combined ~260k tokens)',
    },
}


def add_metrics_section(doc: Document, cat_num: int):
    """Add a 'Key Metrics' section pulling from the generated JSON."""
    import json
    metrics_path = os.path.join(ROOT, 'outputs', 'metrics', f'cat{cat_num}_{_get_metric_suffix(cat_num)}.json')
    if not os.path.exists(metrics_path):
        return

    try:
        with open(metrics_path) as f:
            data = json.load(f)

        add_section_heading(doc, 'Key Computed Metrics', level=1)
        add_body_paragraph(doc, 'The following metrics were computed during experiment execution:')

        _render_metrics_dict(doc, data, depth=0)
    except Exception:
        pass


def _get_metric_suffix(cat_num: int) -> str:
    suffixes = {
        1: 'preprocessing', 2: 'entropy_perplexity', 3: 'rule_morphology',
        4: 'fsa_morphology', 5: 'unigram_model', 6: 'bigram_model',
        7: 'neural_models', 8: 'vector_semantics', 9: 'word_embeddings',
    }
    return suffixes.get(cat_num, '')


def _render_metrics_dict(doc: Document, data, depth: int = 0, prefix: str = ''):
    """Recursively render a nested metrics dict as bullets/tables."""
    if isinstance(data, dict):
        for key, val in list(data.items())[:20]:  # cap at 20 keys
            label = f"{prefix}{key}" if prefix else key
            if isinstance(val, dict):
                add_section_heading(doc, label.replace('_', ' ').title(), level=3)
                _render_metrics_dict(doc, val, depth + 1)
            elif isinstance(val, list) and val and isinstance(val[0], (int, float, str)):
                add_bullet(doc, f"{label.replace('_',' ')}: {val[:5]}")
            elif isinstance(val, (int, float, str, bool)):
                add_bullet(doc, f"**{label.replace('_',' ')}**: {val}")
    elif isinstance(data, list):
        for item in data[:10]:
            if isinstance(item, dict):
                _render_metrics_dict(doc, item, depth + 1)
            else:
                add_bullet(doc, str(item))


def add_output_plots_section(doc: Document, cat_num: int):
    """List all generated plots for this category."""
    plots_dir = os.path.join(ROOT, 'outputs', 'plots')
    cat_plots  = sorted(f for f in os.listdir(plots_dir) if f.startswith(f'cat{cat_num}_') and f.endswith('.png'))
    if not cat_plots:
        return

    add_section_heading(doc, 'Generated Visualizations', level=1)
    for plot_name in cat_plots:
        label = plot_name.replace(f'cat{cat_num}_', '').replace('_', ' ').replace('.png', '').title()
        add_bullet(doc, f"`{plot_name}` — {label}")


def add_footer_info(doc: Document, cat_num: int):
    """Add execution environment info."""
    meta = CATEGORY_META[cat_num]
    add_section_heading(doc, 'Execution Information', level=1)

    info_rows = [
        ['Field', 'Value'],
        ['Category', f'Category {cat_num}'],
        ['Topic', meta['title']],
        ['Datasets Used', meta['datasets']],
        ['Language', 'Python 3.11'],
        ['Key Libraries', 'NLTK, gensim, scikit-learn, matplotlib, seaborn, umap-learn'],
        ['Output Path', f'outputs/metrics/cat{cat_num}_*.json, outputs/plots/cat{cat_num}_*.png'],
        ['Source File', f'experiments/cat{cat_num}_{_get_metric_suffix(cat_num)}/run.py'],
    ]
    table = doc.add_table(rows=len(info_rows), cols=2)
    table.style = 'Table Grid'
    for r_idx, (field, val) in enumerate(info_rows):
        row = table.rows[r_idx]
        for c_idx, text in enumerate([field, val]):
            cell = row.cells[c_idx]
            cell.text = ''
            para = cell.paragraphs[0]
            run  = para.add_run(text)
            run.font.name = FONT_BODY
            run.font.size = Pt(10)
            if r_idx == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_background(cell, '1F497D')
            elif c_idx == 0:
                run.font.bold = True
                set_cell_background(cell, 'D9E6F2')
            elif r_idx % 2 == 0:
                set_cell_background(cell, 'F5F9FF')


# ── Main Builder ──────────────────────────────────────────────────────────────

def build_docx(cat_num: int) -> str:
    """Build a styled .docx for a single category."""
    meta     = CATEGORY_META[cat_num]
    md_fname = f'cat{cat_num}_{_get_metric_suffix(cat_num)}.md'
    md_path  = os.path.join(MD_DIR, md_fname)
    out_path = os.path.join(DOCX_DIR, f'cat{cat_num}_{_get_metric_suffix(cat_num)}.docx')

    if not os.path.exists(md_path):
        print(f"  [SKIP] {md_fname} not found")
        return ''

    doc = Document()
    style_document(doc)
    add_cover_header(doc, cat_num, meta['title'], meta['subtitle'])
    md_to_docx(md_path, doc)
    add_metrics_section(doc, cat_num)
    add_output_plots_section(doc, cat_num)
    add_footer_info(doc, cat_num)

    doc.save(out_path)
    return out_path


def main():
    print("=" * 60)
    print("GENERATING WORD (.docx) LAB RECORDS")
    print("=" * 60)
    os.makedirs(DOCX_DIR, exist_ok=True)

    for cat_num in range(1, 10):
        meta     = CATEGORY_META[cat_num]
        out_path = build_docx(cat_num)
        if out_path:
            size_kb = os.path.getsize(out_path) // 1024
            print(f"  Cat{cat_num}: {meta['title']:<40} → {os.path.basename(out_path)}  ({size_kb} KB)")

    print("\n  All .docx files saved to: docs/lab_records/")
    print("=" * 60)


if __name__ == '__main__':
    main()
