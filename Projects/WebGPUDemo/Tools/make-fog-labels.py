#!/usr/bin/env python3
"""Draws the glyph atlas the `fog` scene samples for bottle names and prices — `Resources/fog-labels.png`.

The scene draws its whole shelf inside WebGPU (so the fog has something to blur without reading the
views beneath the canvas, which WebGPU cannot do), and text is the one thing an SDF shader cannot
conjure — so the words come from this atlas: 5 columns × 8 rows of 128×64 cells, names in rows 0–3
(one per shelf), prices in rows 4–7, white on transparent. Re-run after changing the bottle list in
`DemoSrc/src/fog/index.tsx`; it needs Pillow (`pip3 install pillow`) and a macOS Arial Bold.

    python3 Projects/WebGPUDemo/Tools/make-fog-labels.py
"""
from PIL import Image, ImageDraw, ImageFont
import os

SHELVES = [
    [("COLA", "1,250"), ("SODA", "1,150"), ("LEMON", "1,300"), ("GRAPE", "1,400"), ("CIDER", "1,100")],
    [("PEACH", "1,500"), ("MINT", "1,350"), ("BERRY", "1,600"), ("MELON", "1,250"), ("COCOA", "1,800")],
    [("ORANGE", "1,300"), ("LIME", "1,150"), ("OCEAN", "1,700"), ("PLUM", "1,450"), ("MILK", "1,000")],
    [("CHERRY", "1,350"), ("KIWI", "1,250"), ("SKY", "1,200"), ("HONEY", "1,550"), ("VIOLET", "1,650")],
]
CELL_W, CELL_H = 128, 64
image = Image.new("RGBA", (CELL_W * 5, CELL_H * 8), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
name_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
price_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)

def put(text, column, row, font):
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    x = column * CELL_W + (CELL_W - width) / 2 - box[0]
    y = row * CELL_H + (CELL_H - height) / 2 - box[1]
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

for row, shelf in enumerate(SHELVES):
    for column, (name, price) in enumerate(shelf):
        put(name, column, row, name_font)
        put(price, column, row + 4, price_font)

out = os.path.join(os.path.dirname(__file__), "..", "Resources", "fog-labels.png")
image.save(out, optimize=True)
print("wrote", os.path.normpath(out), image.size)
