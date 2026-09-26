#!/usr/bin/env python3
"""Build 36h11 shoulder tags and optional 14 cm labeled print cards."""
from pathlib import Path
import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ROOT / 'Assets' / 'Markers'
PRINT = ROOT / 'Print'
FONT = '/System/Library/Fonts/Supplemental/Arial Black.ttf'


def make(side: str, tag_id: int) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # Match the earlier assets: 80 mm tag plus a 10 mm white quiet zone on each side.
    # A 180-degree rotation matches the AprilTag images previously used in this project.
    tag = np.rot90(cv2.aruco.generateImageMarker(dictionary, tag_id, 800), 2)
    core = Image.new('RGB', (1000, 1000), 'white')
    core.paste(Image.fromarray(tag).convert('RGB'), (100, 100))
    core_path = MARKERS / f'{side} AprilTag 36h11 ID {tag_id}.png'
    core.save(core_path, optimize=True)
    core_b64 = base64.b64encode(core_path.read_bytes()).decode('ascii')
    (PRINT / f'{side.lower()}-shoulder-id-{tag_id}.svg').write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="120mm" height="125mm" viewBox="0 0 120 125">'
        f'<rect width="120" height="125" fill="white"/>'
        f'<image x="10" y="5" width="100" height="100" xlink:href="data:image/png;base64,{core_b64}"/>'
        f'<text x="10" y="117" font-family="Arial" font-size="5">R1 {side.upper()} shoulder · tag36h11 ID {tag_id} · 100 mm image</text>'
        f'</svg>')
    assert core.size == (1000, 1000)
    card = Image.new('RGB', (1400, 1400), 'white')
    draw = ImageDraw.Draw(card)
    draw.rectangle((12, 12, 1387, 1387), outline='black', width=16)
    card.paste(core, (200, 200))
    big = ImageFont.truetype(FONT, 90)
    medium = ImageFont.truetype(FONT, 55)
    draw.rounded_rectangle((34, 35, 168, 168), radius=14, fill='black')
    draw.text((65, 50), str(tag_id), font=big, fill='white')
    draw.text((220, 54), f'R1 {side.upper()}', font=big, fill='black')
    draw.polygon([(1110, 45), (1360, 90), (1110, 158), (1150, 96)], fill='black')
    draw.text((37, 1230), 'ROBOT', font=medium, fill='black')
    draw.text((760, 1230), f'SHOULDER {side[0].upper()}', font=medium, fill='black')
    # Wide, deliberately asymmetric features remain readable at Spectacles distance.
    if side == 'Left':
        draw.ellipse((45, 245, 155, 355), fill='black')
        draw.ellipse((76, 277, 125, 326), fill='white')
        draw.polygon([(35, 490), (170, 420), (138, 620)], fill='black')
        draw.rounded_rectangle((52, 755, 148, 1035), radius=26, fill='black')
        draw.ellipse((290, 1260, 360, 1330), fill='black')
        draw.polygon([(1245, 240), (1360, 300), (1245, 360)], fill='black')
        draw.rectangle((1260, 520, 1362, 566), fill='black')
        draw.ellipse((1250, 770, 1350, 910), fill='black')
        draw.polygon([(1235, 1080), (1360, 1000), (1340, 1160)], fill='black')
    else:
        draw.polygon([(45, 260), (165, 230), (95, 365)], fill='black')
        draw.rectangle((47, 485, 155, 545), fill='black')
        draw.ellipse((45, 720, 160, 865), fill='black')
        draw.polygon([(40, 1080), (165, 995), (130, 1175)], fill='black')
        draw.polygon([(1240, 225), (1365, 250), (1280, 390)], fill='black')
        draw.rounded_rectangle((1260, 535, 1355, 750), radius=35, fill='black')
        draw.ellipse((1255, 880, 1360, 990), fill='black')
        draw.ellipse((1280, 905, 1335, 965), fill='white')
        draw.rectangle((1260, 1090, 1360, 1160), fill='black')
    name = f'{side} Shoulder Tracking Card'
    png = MARKERS / f'{name}.png'
    card.save(png, optimize=True)
    b64 = base64.b64encode(png.read_bytes()).decode('ascii')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="160mm" height="175mm" viewBox="0 0 160 175">
  <rect width="160" height="175" fill="white"/>
  <image x="10" y="10" width="140" height="140" xlink:href="data:image/png;base64,{b64}"/>
  <text x="80" y="161" text-anchor="middle" font-family="Arial" font-size="5">R1 {side.upper()} SHOULDER · APRILTAG 36h11 ID {tag_id} · PRINT 100%</text>
  <text x="80" y="169" text-anchor="middle" font-family="Arial" font-size="4">Card 140 mm · inner AprilTag 100 mm · upright/front-facing</text>
</svg>'''
    (PRINT / f'{side.lower()}-shoulder-id-{tag_id}-tracking-card.svg').write_text(svg)
    print(png)


make('Left', 22)
make('Right', 23)
