"""Merge annual transition panels and re-layout the original burden chart.

Sankey widths and labels come from formal city-year data. Burden markers and
uncertainty intervals are preserved as source-image strips, not digitized data.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.lines import Line2D
from PIL import Image

OUT = Path(__file__).resolve().parent
FIG = OUT.parent
DATA = OUT / 'source_data'
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12,
                     'text.color': '#202428', 'axes.labelcolor': '#202428',
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})

def sankey():
    source = Path(r'E:\transfer justice\results_v10_reconstructed_2019_2027\data\SPRI_city_annual_2019_2025.csv')
    risk = pd.read_csv(source)
    codes = {'low_current_low_momentum': 'L-S', 'emerging': 'L-G',
             'established_stable_declining': 'H-S', 'established_rising': 'H-G'}
    order = ['L-S', 'L-G', 'H-S', 'H-G']
    colors = dict(zip(order, ['#7893A8', '#719E87', '#9A7EB0', '#BE7A6C']))
    names = {'L-S': 'Low current\nLow momentum', 'L-G': 'Low current\nHigh momentum',
             'H-S': 'High current\nLow momentum', 'H-G': 'High current\nHigh momentum'}
    risk['state'] = risk.quadrant.map(codes)
    assert risk.state.notna().all()
    wide = risk.pivot(index='node_id', columns='year', values='state')
    pairs = []
    for year in range(2019, 2025):
        part = wide[[year, year+1]].dropna().set_axis(['from', 'to'], axis=1)
        part['from_year'] = year
        pairs.append(part.reset_index())
    transitions = pd.concat(pairs, ignore_index=True)
    counts = pd.crosstab(transitions['from'], transitions['to']).reindex(index=order, columns=order, fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0)
    total = int(counts.to_numpy().sum())
    assert total == 8550 and len(wide) == 1425
    transitions.to_csv(DATA/'fig5be_city_year_transitions.csv', index=False)
    records = [{'from': a, 'to': b, 'count': int(counts.loc[a,b]),
                'probability_percent': float(probs.loc[a,b]*100)} for a in order for b in order]
    pd.DataFrame(records).to_csv(DATA/'fig5be_transition_counts_probabilities.csv', index=False)

    fig = plt.figure(figsize=(12.5, 7.1), facecolor='white')
    ax = fig.add_axes([0.015, 0.11, 0.97, 0.79])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    gap = .055
    scale = (1 - 3*gap)/total
    def nodes(totals):
        top = 1.; result = {}
        for state in order:
            h = totals[state]*scale
            result[state] = (top-h, top)
            top -= h+gap
        return result
    left = nodes(counts.sum(axis=1)); right = nodes(counts.sum(axis=0))
    outgoing = {}; incoming = {}
    for a in order:
        top = left[a][1]
        for b in order:
            h = counts.loc[a,b]*scale
            outgoing[a,b] = (top-h, top); top -= h
    for b in order:
        top = right[b][1]
        for a in order:
            h = counts.loc[a,b]*scale
            incoming[a,b] = (top-h, top); top -= h
    x0, x1 = .195, .81
    for a in order:
        for b in order:
            lo, hi = outgoing[a,b]; rlo, rhi = incoming[a,b]
            verts = [(x0,lo),(.49,lo),(.52,rlo),(x1,rlo),
                     (x1,rhi),(.52,rhi),(.49,hi),(x0,hi),(x0,lo)]
            patch = PathPatch(MPath(verts,[MPath.MOVETO]+[MPath.CURVE4]*3+[MPath.LINETO]+[MPath.CURVE4]*3+[MPath.CLOSEPOLY]),
                              facecolor=colors[a], edgecolor='white', lw=.45, alpha=.43 if a != b else .65)
            ax.add_patch(patch)
    for side, pos, totals in [(left,.18,counts.sum(axis=1)),(right,.81,counts.sum(axis=0))]:
        for state, (lo,hi) in side.items():
            ax.add_patch(Rectangle((pos,lo),.015,hi-lo,color=colors[state],lw=0))
            isleft = pos < .5
            x = pos-.012 if isleft else pos+.029
            align = 'right' if isleft else 'left'
            mid = (lo+hi)/2
            ax.text(x,mid+.046,state,ha=align,va='center',fontsize=15,fontweight='bold',color=colors[state])
            ax.text(x,mid-.002,names[state],ha=align,va='center',fontsize=10.5,linespacing=1.12)
            ax.text(x,mid-.050,f'n = {totals[state]:,}',ha=align,va='center',fontsize=10,color='#626B73')
    # Equally spaced callouts keep all 16 numbers readable, including narrow
    # cross-state flows. Each hairline points to its actual ribbon midpoint.
    for a in order:
        lo, hi = left[a]
        label_ys = np.linspace(hi-.024, lo+.024, 4)
        for b, ytext in zip(order,label_ys):
            flowmid = np.mean(outgoing[a,b])
            ax.plot([x0+.003,.226,.244],[flowmid,flowmid,ytext],color=colors[a],lw=.65,zorder=5)
            ax.text(.247,ytext,f'{b}  {counts.loc[a,b]:,} ({100*probs.loc[a,b]:.1f}%)',
                    va='center',ha='left',fontsize=10.5,zorder=6,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.94,pad=1.5))
    fig.text(.19,.945,'State in year t',ha='center',fontsize=15,fontweight='bold')
    fig.text(.81,.945,'State in year t + 1',ha='center',fontsize=15,fontweight='bold')
    fig.text(.5,.043,'2019–2025  ·  1,425 cities  ·  8,550 annual transitions',ha='center',fontsize=11)
    fig.text(.5,.016,'Ribbon width = transition count; labels = destination, count (probability conditional on the origin state).',
             ha='center',fontsize=10,color='#626B73')
    stem = OUT/'fig5be_annotated_state_transition_sankey'
    for ext in ['png','pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=450,facecolor='white')
    fig.savefig(DATA/'fig5be_preview.png',dpi=110,facecolor='white')
    plt.close(fig)
    return {'source':str(source),'cities':1425,'transitions':total,'years':[2019,2025],
            'conditional_probability_denominator':'All outgoing transitions from the origin state',
            'all_16_transitions_labeled':True,'size_inches':[12.5,7.1]}

def landscape_burden():
    source = FIG/'fig4f_regional_disproportionality.png'
    original = Image.open(source).convert('RGB')
    assert original.size == (3217,1641)
    target = Image.open(FIG/'fig4c_gdp_risk_gradient.png')
    width, height = target.size
    dpi = 330
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi))
    fig.subplots_adjust(left=.19,right=.975,top=.82,bottom=.17)
    xmin,xmax=.43,1.66
    ax.set_xlim(xmin,xmax); ax.set_ylim(-.5,6.5)
    ax.axvspan(xmin,1,color='#F0F5F8',zorder=0)
    ax.axvspan(1,xmax,color='#FBF4F4',zorder=0)
    for x in [.6,.8,1.2,1.4,1.6]:
        ax.axvline(x,color='#DDE2E6',lw=.7,zorder=0)
    ax.axvline(1,color='#71787E',ls='--',lw=1.1,zorder=1)
    rows = ['Other','North America','Europe','South America','Africa','Southeast Asia','East Asia']
    emissions = [1.46,.71,1.13,.65,.54,.96,1.04]
    risks = [1.13,.81,.84,1.12,1.22,1.11,.85]
    centers = [158,343,528,712,897,1082,1267]
    # Source chart x coordinates map to the unchanged axis limits. Each strip
    # contains only marks, connecting segments, and uncertainty bars, not labels.
    # Physical aspect is preserved when embedding each raster strip.
    fig.canvas.draw()
    axes_width_in = ax.get_position().width * fig.get_figwidth()
    axes_height_in = ax.get_position().height * fig.get_figheight()
    strip_height_in = axes_width_in * 44/1427
    yhalf = strip_height_in/axes_height_in*7/2
    for i,cy in enumerate(centers):
        strip = original.crop((1758,cy-22,3185,cy+22))
        if i == 6:
            # Original legend overlaps the unused right end of this row.
            # Truncate after the actual E marker; retain both actual markers.
            strip = strip.crop((0,0,2490-1758,44))
        # Make only the source background/grid transparent. Retain the source
        # RGB pixels of blue/red marks, uncertainty bars, and grey connectors.
        pixels = np.array(strip)
        r,g,b = [pixels[:,:,k].astype(float) for k in range(3)]
        blue = (b-r>25) & (b-g>10) & (r<190)
        red = (r-g>22) & (r-b>20)
        connector = (b-r>6) & (g-r>3) & (r<200) & (b<215)
        alpha = ((blue | red | connector)*255).astype(np.uint8)
        strip = Image.fromarray(np.dstack([pixels,alpha]),'RGBA')
        rightx = xmin+(xmax-xmin)*strip.width/1427
        y = 6-i
        ax.imshow(strip,extent=(xmin,rightx,y-yhalf,y+yhalf),aspect='auto',interpolation='none',zorder=3)
        ax.text(emissions[i],y-.27,f'E {emissions[i]:.2f}',ha='center',va='center',fontsize=11,color='#427FAE')
        ax.text(risks[i],y+.27,f'R {risks[i]:.2f}',ha='center',va='center',fontsize=11,color='#B74742',fontweight='bold')
    ax.set_yticks(range(6,-1,-1),rows)
    ax.tick_params(axis='y',length=0,pad=8,labelsize=12)
    ax.set_xticks([.6,.8,1,1.2,1.4,1.6])
    ax.tick_params(axis='x',labelsize=11)
    ax.set_xlabel('Disproportionality ratio (1 = proportional share)',labelpad=8,fontsize=12)
    for side in ['top','right','left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#5C646B')
    handles=[Line2D([],[],marker='D',color='#4F87B7',ls='',markersize=6,label='Emission / population'),
             Line2D([],[],marker='o',color='#CC5C57',ls='',markersize=6,label='Risk burden / population')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.58,.945),ncol=2,frameon=False,fontsize=11,handletextpad=.4,columnspacing=2)
    stem=OUT/'fig4d_burden_relative_to_population'
    fig.savefig(stem.with_suffix('.png'),dpi=dpi,facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'),dpi=dpi,facecolor='white')
    fig.savefig(DATA/'fig4d_burden_relative_to_population_preview.png',dpi=120,facecolor='white')
    plt.close(fig)
    assert Image.open(stem.with_suffix('.png')).size == (width,height)
    return {'source':str(source),'size_pixels':[width,height],'dpi':dpi,
            'reference_size':str(FIG/'fig4c_gdp_risk_gradient.png'),
            'method':'Re-layout with source raster marks and uncertainty bars at preserved aspect; new text and axes. No uncertainty values inferred.',
            'pdf_type':'Vector text/axes with embedded original raster data marks',
            'supplementary_left_panel':'Unchanged'}

if __name__ == '__main__':
    result={'fig5be':sankey(),'fig4d':landscape_burden()}
    (DATA/'sankey_landscape_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
