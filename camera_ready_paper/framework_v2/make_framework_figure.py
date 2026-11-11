#!/usr/bin/env python3
"""Revised SPC-Net framework figure (camera-ready, 2026-11).

Draws the FINAL model only: prompt bank + max aggregation, PPG with adaptive
fallback, SAM block (image-to-token), VLF, U-Net decoder with BCE. No prompt
residual, no attention aggregation. One layout spec -> SVG/HTML/PNG + PPTX slide.

Run: PYTHONPATH=/tmp/pyx python make_framework_figure.py
"""
import base64, copy, os
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1600, 770
FONT = 'Calibri'

# ── Layout spec (pixel units on a W x H canvas) ──────────────────────────────
# kind: box | trap | img | container
BOXES = [
    dict(id='img', kind='img', x=20, y=75, w=280, h=186, src='input.png'),
    dict(id='vit', kind='trap', x=330, y=55, w=250, h=215, fill='#F4B183', stroke='#C55A11',
         title='CLIP ViT-L/14', sub=['fine-tuned', 'hooks: blocks 5, 11, 17, 23']),
    dict(id='fv', kind='box', x=620, y=95, w=230, h=105, fill='#FBE5D6', stroke='#C55A11',
         title='Visual features F_v', sub=['high · mid · skips']),
    dict(id='bank', kind='box', x=120, y=300, w=180, h=115, fill='#F2F0EB', stroke='#595959',
         title='Prompt bank', sub=['K = 10 salient-object', 'descriptions'], star=True),
    dict(id='txt', kind='trap', x=330, y=290, w=250, h=135, fill='#A9D18E', stroke='#548235',
         title='CLIP text encoder', sub=['frozen']),
    dict(id='fl', kind='box', x=620, y=305, w=230, h=105, fill='#E2F0D9', stroke='#548235',
         title='Text embeddings', sub=['K prompt vectors']),
    dict(id='corr', kind='box', x=900, y=185, w=280, h=125, fill='#EDE7F6', stroke='#5B4BB7',
         title='V-L correlation C', sub=['cosine similarity to K prompts', 'max over prompts'], star=True),
    dict(id='dec', kind='box', x=1265, y=300, w=300, h=120, fill='#FFF2CC', stroke='#5B4BB7',
         title='Mask decoder', sub=['U-Net with CLIP skips', 'BCE loss']),
    dict(id='mask', kind='img', x=1295, y=475, w=240, h=160, src='mask.png'),
    dict(id='esc', kind='container', x=95, y=470, w=1110, h=230),
    dict(id='ppg', kind='box', x=125, y=530, w=290, h=125, fill='#F2F0EB', stroke='#595959',
         title='Pseudo prompt generator', sub=['softmax → k-means → points + masks', 'adaptive fallback (top-ρ)'], star=True),
    dict(id='sam', kind='box', x=465, y=530, w=250, h=125, fill='#B4C7E7', stroke='#2F5597',
         title='SAM block', sub=['image-to-token cross-attn', 'refines F_v (high)']),
    dict(id='vlf', kind='box', x=765, y=530, w=280, h=125, fill='#BDD7EE', stroke='#2F5597',
         title='VLF block', sub=['Swin window attention', '+ text cross-attention']),
]
LABELS = [  # (x, y, text, size, bold, anchor, color)
    (160, 60, 'Input image', 22, True, 'middle', '#000000'),
    (1415, 668, 'Predicted saliency mask', 20, True, 'middle', '#000000'),
    (1415, 692, 'M (1 × H × W)', 15, False, 'middle', '#000000'),
    (118, 500, 'ESC block × N', 20, True, 'start', '#000000'),
    (1075, 28, 'CLIP skip features → decoder', 15, False, 'middle', '#000000'),
    (1048, 432, 'C', 15, True, 'start', '#000000'),
    (440, 578, 'prompts', 12, False, 'middle', '#404040'),
    (740, 578, 'F_v', 12, False, 'middle', '#404040'),
    (585, 690, 'updated C → PPG of the next ESC block', 13, False, 'middle', '#404040'),
    (1125, 583, 'refined F_v', 13, False, 'middle', '#404040'),
    (1125, 620, '(after N blocks)', 12, False, 'middle', '#404040'),
    (125, 745, 'Components we add to ESC-Net', 15, False, 'start', '#000000'),
    (808, 497, 'caption embedding', 13, False, 'start', '#404040'),
    (808, 514, '(single text token)', 12, False, 'start', '#404040'),
]
ARROWS = [  # (points, dashed)
    ([(300, 168), (330, 168)], False),                      # image -> ViT
    ([(580, 148), (620, 148)], False),                      # ViT -> F_v
    ([(300, 357), (330, 357)], False),                      # bank -> text enc
    ([(580, 357), (620, 357)], False),                      # text enc -> F_l
    ([(850, 148), (875, 148), (875, 228), (900, 228)], False),   # F_v -> corr
    ([(850, 357), (875, 357), (875, 268), (900, 268)], False),   # F_l -> corr
    ([(735, 95), (735, 40), (1415, 40), (1415, 300)], False),    # skips -> decoder
    ([(1040, 310), (1040, 445), (270, 445), (270, 530)], False), # C -> PPG
    ([(415, 592), (465, 592)], False),                      # PPG -> SAM
    ([(715, 592), (765, 592)], False),                      # SAM -> VLF
    ([(905, 655), (905, 675), (270, 675), (270, 655)], True),    # VLF -> next PPG
    ([(1045, 600), (1235, 600), (1235, 395), (1265, 395)], False),  # refined F_v -> decoder
    ([(1415, 420), (1415, 475)], False),                    # decoder -> mask
    ([(800, 410), (800, 530)], False),                      # caption embedding -> VLF
]


import math


def star_pts(cx, cy, r):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def star_svg(cx, cy, r):
    return '<polygon fill="#000" points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in star_pts(cx, cy, r)) + '"/>'


def sub_html(t, size=16):
    """F_v -> F + U+1D65 (subscript v); F_l -> F + smaller l (no baseline shift:
    cairosvg mispositions shifted tspans in centred text)."""
    t = t.replace('F_v', 'F\u1d65')
    return t.replace('F_l', f'F<tspan font-size="{size*0.7:.0f}">l</tspan>')


# ── SVG ──────────────────────────────────────────────────────────────────────
def svg():
    E = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
         f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Calibri, Carlito, Helvetica, Arial, sans-serif">',
         '<defs><marker id="ah" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
         '<path d="M0,0 L10,4 L0,8 z" fill="#000"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for b in BOXES:
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        if b['kind'] == 'img':
            data = base64.b64encode(open(os.path.join(HERE, b['src']), 'rb').read()).decode()
            E.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" '
                     f'xlink:href="data:image/png;base64,{data}"/>')
            continue
        if b['kind'] == 'container':
            E.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" '
                     f'stroke="#000" stroke-width="2" stroke-dasharray="9,7"/>')
            continue
        if b['kind'] == 'trap':
            d = h * 0.18
            E.append(f'<polygon points="{x},{y} {x+w},{y+d} {x+w},{y+h-d} {x},{y+h}" '
                     f'fill="{b["fill"]}" stroke="{b["stroke"]}" stroke-width="2.5"/>')
        else:
            E.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{b["fill"]}" '
                     f'stroke="{b["stroke"]}" stroke-width="2.5"/>')
        n = len(b['sub'])
        cy = y + h / 2 - (n * 19) / 2 + 4
        E.append(f'<text x="{x+w/2}" y="{cy}" text-anchor="middle" font-size="23" font-weight="bold">'
                 f'{sub_html(b["title"], 23)}</text>')
        for i, s in enumerate(b['sub']):
            E.append(f'<text x="{x+w/2}" y="{cy + 26 + i*21}" text-anchor="middle" font-size="16">'
                     f'{sub_html(s, 16)}</text>')
        if b.get('star'):
            E.append(star_svg(x + 16, y + 17, 10))
    for pts, dashed in ARROWS:
        p = ' '.join(f'{a},{c}' for a, c in pts)
        E.append(f'<polyline points="{p}" fill="none" stroke="#000" stroke-width="2.2" '
                 f'marker-end="url(#ah)"' + (' stroke-dasharray="7,5"' if dashed else '') + '/>')
    for x, y, t, sz, bold, anc, col in LABELS:
        E.append(f'<text x="{x}" y="{y}" text-anchor="{anc}" font-size="{sz}" fill="{col}"'
                 + (' font-weight="bold"' if bold else '') + f'>{sub_html(t, sz)}</text>')
    E.append(star_svg(110, 739, 11))
    E.append('</svg>')
    return '\n'.join(E)


# ── PPTX slide (editable shapes) ─────────────────────────────────────────────
def rgb(h):
    return RGBColor.from_string(h.lstrip('#'))


def add_arrowhead(line_elm_owner):
    ln = line_elm_owner.line._get_or_add_ln()
    tail = ln.find(qn('a:tailEnd'))
    if tail is None:
        tail = ln.makeelement(qn('a:tailEnd'), {})
        ln.append(tail)
    tail.set('type', 'triangle'); tail.set('w', 'med'); tail.set('len', 'med')


def add_text(tf, lines, align=PP_ALIGN.CENTER):
    """lines: list of (text, size_pt, bold, color)."""
    tf.clear(); tf.word_wrap = True
    for i, (t, sz, bold, col) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        parts = t.split('F_')
        runs = [(parts[0], False)] + [x for q in parts[1:] for x in (('F', False), (q[0], True), (q[1:], False))]
        for txt, is_sub in runs:
            if not txt:
                continue
            r = p.add_run(); r.text = txt
            r.font.size = Pt(sz * (0.7 if is_sub else 1)); r.font.bold = bold
            r.font.name = FONT; r.font.color.rgb = rgb(col)
            if is_sub:
                r.font._element.set('baseline', '-25000')


def add_star(sl, x, y, d):
    st_ = sl.shapes.add_shape(MSO_SHAPE.STAR_5_POINT, x, y, d, d)
    st_.fill.solid(); st_.fill.fore_color.rgb = rgb('#000000'); st_.line.fill.background()
    st_.shadow.inherit = False; st_.name = 'star'


def pptx_slide(src, dst):
    prs = Presentation(src)
    sw, sh = prs.slide_width, prs.slide_height
    s = min(sw / W, sh / H)
    ox, oy = (sw - W * s) / 2, (sh - H * s) / 2
    X = lambda v: Emu(int(ox + v * s)); Y = lambda v: Emu(int(oy + v * s)); L = lambda v: Emu(int(v * s))
    pt = lambda px: px * s / 12700 * 0.95   # px -> pt at this scale
    blank = min(prs.slide_layouts, key=lambda l: len(l.placeholders))
    sl = prs.slides.add_slide(blank)
    for b in BOXES:
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        if b['kind'] == 'img':
            sl.shapes.add_picture(os.path.join(HERE, b['src']), X(x), Y(y), L(w), L(h)); continue
        if b['kind'] == 'container':
            c = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(x), Y(y), L(w), L(h))
            c.adjustments[0] = 0.06; c.fill.background()
            c.line.color.rgb = rgb('#000000'); c.line.width = Pt(1.5)
            c.line.dash_style = MSO_LINE_DASH_STYLE.DASH; c.name = 'ESC container'; continue
        if b['kind'] == 'trap':
            d = h * 0.18
            fb = sl.shapes.build_freeform(X(x), Y(y), scale=1.0)
            fb.add_line_segments([(X(x + w), Y(y + d)), (X(x + w), Y(y + h - d)), (X(x), Y(y + h))], close=True)
            shp = fb.convert_to_shape()
        else:
            shp = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(x), Y(y), L(w), L(h))
            shp.adjustments[0] = 0.1
        shp.name = b['id']
        shp.fill.solid(); shp.fill.fore_color.rgb = rgb(b['fill'])
        shp.line.color.rgb = rgb(b['stroke']); shp.line.width = Pt(2)
        shp.shadow.inherit = False
        tb = sl.shapes.add_textbox(X(x), Y(y), L(w), L(h)); tb.name = b['id'] + ' text'
        tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        add_text(tb.text_frame, [(b['title'], pt(23), True, '#000000')] +
                 [(t, pt(16), False, '#000000') for t in b['sub']])
        if b.get('star'):
            add_star(sl, X(x + 6), Y(y + 7), L(20))
    for pts, dashed in ARROWS:
        fb = sl.shapes.build_freeform(X(pts[0][0]), Y(pts[0][1]), scale=1.0)
        fb.add_line_segments([(X(a), Y(c)) for a, c in pts[1:]], close=False)
        a = fb.convert_to_shape(); a.fill.background(); a.name = 'arrow'
        a.line.color.rgb = rgb('#000000'); a.line.width = Pt(1.75); a.shadow.inherit = False
        if dashed:
            a.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        add_arrowhead(a)
    for x, y, t, sz, bold, anc, col in LABELS:
        bw = 520
        bx = x - bw / 2 if anc == 'middle' else x
        tb = sl.shapes.add_textbox(X(bx), Y(y - sz * 1.15), L(bw), L(sz * 1.6))
        add_text(tb.text_frame, [(t, pt(sz), bold, col)],
                 PP_ALIGN.CENTER if anc == 'middle' else PP_ALIGN.LEFT)
        tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
    add_star(sl, X(98), Y(727), L(23))
    prs.save(dst)
    return len(prs.slides)


if __name__ == '__main__':
    import cairosvg
    S = svg()
    open(os.path.join(HERE, 'SOD_framework_v2.svg'), 'w').write(S)
    open(os.path.join(HERE, 'SOD_framework_v2.html'), 'w').write(
        '<!doctype html><html><head><meta charset="utf-8"><title>SPC-Net framework (camera-ready)</title>'
        '<style>body{margin:20px;background:#fafafa}svg{background:#fff;max-width:100%;height:auto;'
        'box-shadow:0 0 6px #ccc}</style></head><body>\n' + S + '\n</body></html>')
    cairosvg.svg2png(bytestring=S.encode(), write_to=os.path.join(HERE, 'SOD_framework_v2.png'), scale=2)
    n = pptx_slide(os.path.join(HERE, '..', 'SOD_framework.pptx'),
                   os.path.join(HERE, 'SOD_framework_v2.pptx'))
    print('wrote SVG/HTML/PNG; pptx slides:', n)
