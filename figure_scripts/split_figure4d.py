"""Losslessly extract the two scientific chart panels; do not recalculate data."""
from pathlib import Path
import json
from PIL import Image

ROOT=Path(r'E:\transfer justice\manuscript_materials_nature_sustainability')
OUT=Path(__file__).resolve().parent
SUP=ROOT/'04_extended_data/inequality'
SRC=ROOT/'02_results_dynamic_risk/figures/fig4f_regional_disproportionality.png'
im=Image.open(SRC).convert('RGB')
assert im.size==(3217,1641), 'Reinspect extraction coordinates if source changes.'
DPI=330

# Keep the complete right chart, including its region labels and E/R legend.
# White gutters remove only the adjacent left chart's edge and 100% tick.
right=im.crop((1390,0,3217,1462))
right.paste('white',(0,0,6,1462))
right.paste('white',(0,1318,83,1462))
main=Image.new('RGB',(right.width+48,right.height+36),'white')
main.paste(right,(24,18))
main.save(OUT/'fig4d_burden_relative_to_population.png',dpi=(DPI,DPI))
main.save(OUT/'fig4d_burden_relative_to_population.pdf',resolution=DPI)

# Keep the left chart's original pixels and arrange the original seven legend
# entries in two columns below it, rather than retaining the composite legend.
left=im.crop((0,0,1390,1462))
legend_boxes=[
    (925,1467,1367,1535), # East Asia
    (925,1540,1367,1610), # Southeast Asia
    (1380,1467,1810,1535), # Africa
    (1380,1540,1810,1610), # South America
    (1840,1467,2290,1535), # Europe
    (1840,1540,2290,1610), # North America
    (2300,1467,2580,1535), # Other
]
supp=Image.new('RGB',(1503,1830),'white')
supp.paste(left,(24,18))
# The 100% tick extends past the plot edge. Preserve its complete original
# pixels without introducing any labels from the neighbouring right chart.
supp.paste(im.crop((0,1318,1455,1462)),(24,1336))
for i,box in enumerate(legend_boxes):
    item=im.crop(box)
    col=i%2; row=i//2
    supp.paste(item,(175+col*590,1490+row*78))
supp.save(SUP/'figS_regional_global_total_composition.png',dpi=(DPI,DPI))
supp.save(SUP/'figS_regional_global_total_composition.pdf',resolution=DPI)

# PDF outputs embed the exact raster panels; no claim of vector reconstruction.
preview=OUT/'source_data'
preview.mkdir(exist_ok=True)
for picture,name in [(main,'fig4d_burden_relative_to_population_preview.png'),
                     (supp,'figS_regional_global_total_composition_preview.png')]:
    picture.thumbnail((1400,1400))
    picture.save(preview/name)
(preview/'fig4d_split_manifest.json').write_text(json.dumps({
    'source':str(SRC),'source_size':list(im.size),'dpi':DPI,
    'main':'fig4d_burden_relative_to_population.png',
    'supplementary':str(SUP/'figS_regional_global_total_composition.png'),
    'method':'Original chart pixel extraction; all chart data and uncertainty intervals retained.',
    'pdf_type':'Raster-embedded PDF',
},indent=2),encoding='utf-8')
print('Extracted main right panel and supplementary left panel.')
