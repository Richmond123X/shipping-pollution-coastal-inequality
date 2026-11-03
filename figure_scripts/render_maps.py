"""Reproducible redraws of manuscript panels 4a, 5c and 5d from formal data."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge, Circle
from matplotlib.offsetbox import DrawingArea, AnnotationBbox
import cartopy.crs as ccrs
import cartopy.feature as cf

OUT = Path(__file__).resolve().parent
ROOT = Path(r'E:\transfer justice')
DATA = ROOT / 'results_v10_reconstructed_2019_2027/data'
SOURCE = OUT / 'source_data'
SOURCE.mkdir(exist_ok=True)
INK = '#202428'
COMP = ['#7FB5D5', '#C7CDD3', '#EDA6A6']
STATES = {
    'low_current_low_momentum': ('Low current / low momentum', '#586F89'),
    'established_stable_declining': ('High current / low momentum', '#7545A5'),
    'emerging': ('Low current / high momentum', '#078256'),
    'established_rising': ('High current / high momentum', '#C43744'),
}
SHADE_COLORS = {
    'low_current_low_momentum': ['#DCE5ED', '#A5B8CA', '#6A879F', '#294C68'],
    'established_stable_declining': ['#E8DEEE', '#C1A4CE', '#9468A5', '#593969'],
    'emerging': ['#DFEBE3', '#ACC9B6', '#709D83', '#315F45'],
    'established_rising': ['#F2DED8', '#DCAFA3', '#B77668', '#82483F'],
}
RISK_SHADE_BOUNDS = [0, .4, .5, .6, 1.]
RANK_BOUNDS = [-1300, -600, -300, -100, -30, 30, 100, 300, 600, 1300]
RANK_COLORS = ['#274B6B', '#4F7F9F', '#8FB4CA', '#CBDFE8', '#E9EDEF',
               '#F0D7D0', '#DDA493', '#BA7060', '#83483F']
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
    'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
RISK = pd.read_csv(DATA / 'SPRI_city_annual_2019_2025.csv')
LATEST = RISK.loc[RISK.year.eq(2025)].set_index('node_id').copy()
assert LATEST.index.is_unique and len(LATEST) == 1425
assert np.isfinite(LATEST[['rep_lon', 'rep_lat']].to_numpy()).all()
PC = ccrs.PlateCarree()
PROJ = ccrs.Robinson()
HALO = [pe.withStroke(linewidth=2.6, foreground='white')]

def canvas(title, letter):
    fig = plt.figure(figsize=(12.5, 7.1))
    ax = fig.add_axes([.035, .18, .93, .72], projection=PROJ)
    ax.set_global()
    ax.add_feature(cf.LAND.with_scale('50m'), facecolor='#F1F3F5',
                   edgecolor='#B7C0C8', linewidth=.27, zorder=0)
    ax.add_feature(cf.BORDERS.with_scale('50m'), edgecolor='#C6CDD3',
                   linewidth=.22, zorder=.1)
    ax.spines['geo'].set_edgecolor('#88949E')
    ax.spines['geo'].set_linewidth(.65)
    fig.text(.035, .945, f'({letter})', weight='bold', fontsize=16)
    fig.text(.08, .945, title, fontsize=15, va='baseline')
    return fig, ax

def save(fig, name):
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{suffix}', dpi=450)
    fig.savefig(SOURCE / f'{name}_preview.png', dpi=140)
    plt.close(fig)

def label(ax, row, text, dx, dy, color=INK, size=16):
    point = PROJ.transform_point(row.rep_lon, row.rep_lat, PC)
    obj = ax.annotate(text, xy=point, xytext=(dx, dy), textcoords='offset points',
        ha='left' if dx >= 0 else 'right', va='center', fontsize=size, color=color,
        zorder=12, path_effects=HALO,
        arrowprops={'arrowstyle': '-', 'color': '#929DA6', 'lw': .55,
                    'shrinkA': 3, 'shrinkB': 3, 'connectionstyle': 'arc3,rad=0'})
    return obj

def lookup(fragment):
    matches = LATEST.loc[LATEST.index.str.contains(fragment, regex=False)]
    if len(matches) != 1:
        raise ValueError((fragment, matches.index.tolist()))
    return matches.iloc[0], matches.index[0]

def figure4a():
    # E1 is city health-weighted emission exposure. Decompose before final
    # normalization; generic level_component also includes other risk domains.
    manifest = json.loads((DATA / 'risk_method_manifest.json').read_text('utf-8'))
    d = LATEST.copy()
    d['current_contribution'] = d.E1_now_z
    d['history_contribution'] = manifest['mu'] * d.E1_hist_z
    d['future_contribution'] = manifest['lambda'] * d.E1_future_z
    cols = ['current_contribution', 'history_contribution', 'future_contribution']
    d['temporal_total'] = d[cols].sum(axis=1)
    for col in cols:
        d[col.replace('contribution', 'share')] = d[col] / d.temporal_total
    assert (d[cols].to_numpy() >= 0).all()
    assert np.allclose(d[[c.replace('contribution', 'share') for c in cols]].sum(axis=1), 1)
    fig, ax = canvas('Temporal composition of city shipping exposure, 2025', 'a')
    ax.scatter(d.rep_lon, d.rep_lat, s=10, color='#B8C1C9', linewidths=0,
               alpha=.55, transform=PC, zorder=1)
    # Composition-stratified illustrations, with geographic coverage. These
    # examples are not a frequency-representative sample of the city population.
    d['composition_band'] = pd.cut(d.future_share,
        bins=[0,.2,.35,.5,.65,.8,1.00001],right=False,
        labels=['<20%', '20–35%', '35–50%', '50–65%', '65–80%', '80–100%'])
    selection = [
        ('Canada__Greater_Vancouver', 'Vancouver', 0, 0),
        ('United_States__Los_Angeles', 'Los Angeles', 0, 0),
        ('Mexico__Campeche', 'Campeche', 0, 0),
        ('United_States__Harrison', 'Harrison', 0, 0),
        ('United_States__Richmond-Hopewell', 'Richmond–Hopewell', 0, 0),
        ('China__Shanghai', 'Shanghai', 0, 0),
        ('Ukraine__Kherson', 'Kherson', 0, 0),
        ('Venezuela__Anzoátegui', 'Anzoátegui', 0, 0),
        ('Bangladesh__Khulna', 'Khulna', .66, .72),
        ('Morocco__Agadir-Ida_ou_Tanane', 'Agadir', 0, 0),
        ('Indonesia__Merauke', 'Merauke', 0, 0),
        ('Ireland__Galway', 'Galway', 0, 0),
        ('Tanzania__Mtwara', 'Mtwara', 0, 0),
        ('Somalia__Kismaayo', 'Kismayo', 0, 0),
        ('Thailand__Ranong', 'Ranong', .72, .59),
        ('Brazil__Ilhéus', 'Ilhéus', 0, 0),
        ('Indonesia__Poso', 'Poso', .87, .35),
        ('Uruguay__Paysandú', 'Paysandú', 0, 0),
        ('Cambodia__Phnom_Penh', 'Phnom Penh', .86, .65),
        ('Indonesia__Indragiri_Hulu', 'Indragiri Hulu', .75, .32),
        ('Madagascar__Morondava', 'Morondava', 0, 0),
    ]
    rows = []
    # Resolve the three original display names to authoritative administrative nodes.
    replacements = {'Egypt__': "Egypt__Janub_Sina'", 'South_Africa__': 'South_Africa__eThekwini',
                    'Argentina__': 'Argentina__Distrito_Federal'}
    for fragment, name, fx, fy in selection:
        if fragment in replacements:
            candidates = d.loc[d.index.str.contains(replacements[fragment], case=False, regex=False)]
            if len(candidates) != 1:
                raise ValueError((name, d[d.index.str.startswith(fragment)].index.tolist()))
            node = candidates.index[0]
        else:
            _, node = lookup(fragment)
        row = d.loc[node]
        radius = 15.5 * np.sqrt(row.temporal_total / 2.)
        da = DrawingArea(2*radius+2, 2*radius+2, 0, 0)
        angle = 90
        for val, color in zip(row[cols].to_numpy(float) / row.temporal_total, COMP):
            da.add_artist(Wedge((radius+1, radius+1), radius, angle, angle+360*val,
                               facecolor=color, edgecolor='white', linewidth=.55))
            angle += 360*val
        da.add_artist(Circle((radius+1, radius+1), radius, fill=False,
                             edgecolor='#89959F', linewidth=.4))
        point = PROJ.transform_point(row.rep_lon, row.rep_lat, PC)
        displaced = name in {'Khulna', 'Poso', 'Phnom Penh', 'Indragiri Hulu', 'Ranong'}
        local_offset = {'Harrison':(-18,8), 'Campeche':(-8,-7), 'Mtwara':(-20,0)}.get(name,(0,0))
        if displaced or local_offset != (0,0):
            ax.scatter([row.rep_lon], [row.rep_lat], s=18, c=INK, linewidths=0,
                       transform=PC, zorder=3)
        ab = AnnotationBbox(da, point, xybox=(fx, fy) if displaced else local_offset, xycoords='data',
            boxcoords='axes fraction' if displaced else 'offset points', frameon=False, pad=0,
            arrowprops={'arrowstyle':'-', 'color':'#AAB3BA', 'lw':.45} if displaced or local_offset != (0,0) else None,
            zorder=6)
        ax.add_artist(ab)
        side_labels = {'Mtwara', 'Morondava', 'Kismayo', 'Richmond–Hopewell',
                       'Anzoátegui', 'Merauke', 'Harrison', 'Ilhéus'}
        name_offsets = {n:(radius+5,4) for n in side_labels}
        name_offsets['Fiji'] = (-10,-radius-6)
        name_offset = name_offsets.get(name,(0,-radius-6))
        custom_labels = {
            'Vancouver': (-18, radius+18, 'center'),
            'Los Angeles': (-radius-7, 3, 'right'),
            'Campeche': (-8, -radius-9, 'right'),
            'Harrison': (25, radius+28, 'left'),
            'Richmond–Hopewell': (radius+6, 8, 'left'),
            'Galway': (-radius-5, 9, 'right'),
            'Kherson': (0, radius+20, 'center'),
            'Kismayo': (-radius-5, radius+8, 'right'),
            'Mtwara': (-radius-5, 1, 'right'),
            'Morondava': (-radius-6, -10, 'right'),
            'Shanghai': (-radius-5, 18, 'right'),
            'Phnom Penh': (radius+5, -2, 'left'),
            'Ranong': (radius+5, 6, 'left'),
        }
        dx, dy, ha = custom_labels.get(name, (*name_offset, 'left' if name in side_labels else 'center'))
        display = name.replace('Richmond–Hopewell', 'Richmond–\nHopewell')
        ax.annotate(display, xy=(fx, fy) if displaced else point,
                    xycoords='axes fraction' if displaced else 'data',
                    xytext=(dx+local_offset[0],dy+local_offset[1]), textcoords='offset points',
                    ha=ha, va='top', fontsize=16, path_effects=HALO, zorder=8)
        record = row.to_dict(); record.update(node_id=node, display_name=name,
                                             leader_line=displaced or local_offset != (0,0))
        rows.append(record)
    fig.legend(handles=[Patch(facecolor=c, edgecolor='none', label=l) for c,l in
        zip(COMP, ['Current level', 'Historical burden', 'Future momentum'])],
        loc='lower center', bbox_to_anchor=(.36,.065), frameon=False, ncol=3,
        handlelength=1.1, columnspacing=1.1, fontsize=15)
    # A separate size key makes the three-component scale explicit.
    handles = [Line2D([], [], ls='', marker='o', color='#C7CDD3', markeredgecolor='#89959F',
                      markeredgewidth=.4, markersize=31*np.sqrt(v/2), label=f'{v:g}') for v in [.5,1.5,2.5]]
    size_legend = fig.legend(handles=handles, title='Combined temporal score', title_fontsize=15,
               loc='lower center', bbox_to_anchor=(.81,.05), frameon=False,
               ncol=3, handletextpad=.8, columnspacing=1.1, fontsize=14)
    size_legend._legend_box.sep = 15
    selected = pd.DataFrame(rows)
    selected.to_csv(SOURCE/'fig4a_selected_city_components.csv', index=False)
    bands=pd.concat([d.composition_band.value_counts(sort=False).rename('all_city_count'),
                     selected.composition_band.value_counts(sort=False).rename('selected_city_count')],axis=1)
    bands.to_csv(SOURCE/'fig4a_composition_band_coverage.csv',index_label='future_share_band')
    d.to_csv(SOURCE/'fig4a_all_city_components.csv')
    save(fig, 'fig4a_temporal_components_pie_map')

def risk_area(values):
    """A disclosed common power scale expands risk-size contrast."""
    return 150 * (np.asarray(values, dtype=float) / 80.) ** 3

def figure5c(shade=True):
    fig, ax = canvas('City risk profiles and risk magnitude, 2025', 'c')
    if shade:
        ax.set_position([.035,.215,.93,.685])
    # In the adopted equal-size map, draw high-risk cities last to retain
    # their visibility along densely populated coastlines.
    drawing = LATEST.sort_values('SPRI', ascending=shade).copy()
    drawing['symbol_area_pt2'] = 65. if shade else risk_area(drawing.SPRI)
    ramps={key:ListedColormap(SHADE_COLORS[key],name=f'{key}_risk')
           for key,(_,color) in STATES.items()}
    shade_norm=BoundaryNorm(RISK_SHADE_BOUNDS,4,clip=True)
    drawing['risk_shade_class']=pd.cut(drawing.SPRI/100,RISK_SHADE_BOUNDS,
                        right=False,include_lowest=True).astype(str)
    drawing['symbol_color'] = [to_hex(ramps[k](shade_norm(v/100))) for k,v in
                              zip(drawing.quadrant,drawing.SPRI)] if shade else (
                              drawing.quadrant.map({k:v[1] for k,v in STATES.items()}))
    ax.scatter(drawing.rep_lon, drawing.rep_lat, s=drawing.symbol_area_pt2,
               c=drawing.symbol_color,
               edgecolors='#FFFFFF', linewidths=.4, transform=PC, zorder=3)
    zero = drawing[drawing.SPRI.eq(0)]
    if not shade:
        ax.scatter(zero.rep_lon,zero.rep_lat,s=4,marker='+',c='#87939D',
                   linewidths=.5,transform=PC,zorder=4)
    examples = [
        ('Haiti__Haiti','Port-au-Prince',20,48),
        ('Mexico__Ensenada','Ensenada',-20,-33),
        ('Philippines__Metropolitan_Manila','Manila',42,18),
        ('Tanzania__Dar_es_Salaam-Zanzibar_West','Dar es Salaam',35,-15),
        ('China__Shanghai','Shanghai',40,29),
        ('Netherlands__Zuid-Holland','Rotterdam',-28,37),
        ('United_States__Los_Angeles','Los Angeles',-45,22),
        ('Nigeria__Lagos','Lagos',-40,-20),
    ]
    for frag,name,dx,dy in examples:
        row,_ = lookup(frag)
        label(ax,row,name,dx,dy)
    counts = LATEST.quadrant.value_counts()
    handles = [Line2D([], [], ls='', marker='o', markersize=7, markeredgecolor='white',
                    markeredgewidth=.3, color=color, label=f'{name}  (n = {counts[key]:,})')
                    for key,(name,color) in STATES.items()]
    if shade:
        for i,(key,(name,color)) in enumerate(STATES.items()):
            cx=fig.add_axes([.065+i*.24,.10,.185,.016])
            cb=fig.colorbar(ScalarMappable(norm=shade_norm,cmap=ramps[key]),cax=cx,
                            orientation='horizontal',ticks=RISK_SHADE_BOUNDS,
                            spacing='uniform',drawedges=True)
            cx.set_title(name.replace(' / ', '\n'),fontsize=14,pad=8)
            cb.outline.set_linewidth(.45)
            cx.tick_params(labelsize=13,length=2.5,width=.5)
        fig.text(.5,.040,'Full SPRI · four common risk intervals, from light to dark',
                 ha='center',fontsize=15)
    else:
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.40,.045),
                   ncol=2, frameon=False, columnspacing=1.4, labelspacing=.9, fontsize=10)
        size_handles=[Line2D([],[],ls='',marker='o',markersize=np.sqrt(risk_area(v)),
                             color='#A6B0B9',markeredgecolor='white',markeredgewidth=.3,
                             label=f'{v/100:.1f}') for v in [20,50,80]]
        leg=fig.legend(handles=size_handles,title='Full SPRI (cubic area scale)',title_fontsize=9.5,
                       loc='lower center',bbox_to_anchor=(.855,.06),ncol=3,frameon=False,
                       columnspacing=.9,handletextpad=.4,fontsize=10)
        leg._legend_box.sep=9
    drawing[['city','rep_lon','rep_lat','current_level','future_direction','quadrant',
             'SPRI','symbol_area_pt2','symbol_color','risk_shade_class']].to_csv(
        SOURCE/('fig5c_city_risk_profiles_shade_2025.csv' if shade else 'fig5c_city_risk_profiles_2025.csv'))
    save(fig,'fig5c_city_risk_profiles_shade_map' if shade else 'fig5c_city_risk_profiles_map')

def figure5d():
    w = RISK.pivot(index='node_id', columns='year', values='rank_full')
    d = LATEST[['city','rep_lon','rep_lat']].join(w[[2019,2025]])
    # Rank 1 is highest risk. A smaller 2025 rank is a positive upward shift.
    d['rank_rise'] = d[2019] - d[2025]
    assert d.rank_rise.notna().all()
    fig,ax = canvas('Change in city risk ranking, 2019–2025', 'd')
    cmap = ListedColormap(RANK_COLORS,name='rank_change_intervals')
    norm = BoundaryNorm(RANK_BOUNDS,len(RANK_COLORS),clip=True)
    d['rank_change_color_class']=pd.cut(d.rank_rise,RANK_BOUNDS,right=False).astype(str)
    valid=d.sort_values('rank_rise', key=abs)
    sc=ax.scatter(valid.rep_lon,valid.rep_lat,c=valid.rank_rise,cmap=cmap,norm=norm,
                  s=60,edgecolors='white',linewidths=.38,transform=PC,zorder=3)
    examples=[
        ('China__Shanghai','Shanghai',30,39),
        ('Germany__Hamburg','Hamburg',22,53),
        ('Philippines__Metropolitan_Manila','Manila',46,-8),
        ('Tanzania__Dar_es_Salaam-Zanzibar_West','Dar es Salaam',30,-30),
        ('Mozambique__Maputo_City','Maputo',-28,-42),
        ('Netherlands__Zuid-Holland','Rotterdam',-33,37),
        ('United_States__Los_Angeles','Los Angeles',-40,25),
        ('Brazil__Rio_de_Janeiro','Rio de Janeiro',40,-22),
        ('Colombia__Riohacha','Riohacha',-52,25),
        ('Peru__Huaura','Huaura',-50,-5),
        ('Brazil__Pelotas','Pelotas',25,-60),
        ('Uruguay__Montevideo','Montevideo',-38,-43),
        ('United_Arab_Emirates__Dubai','Dubai',25,24),
        ('Myanmar__Mon','Mon',-40,-5),
        ('Japan__Fukuoka','Fukuoka',40,-13),
    ]
    annotations=[]
    for frag,name,dx,dy in examples:
        _,node=lookup(frag); row=d.loc[node]; value=row.rank_rise
        if value == 0:
            continue
        direction='↑' if value>0 else '↓'
        text=f'{name}\n{direction}{abs(value):g}'
        label(ax,row,text,dx,dy,color='#A44642' if value>0 else '#3D6B99' if value<0 else '#66717A',size=16)
        annotations.append({'node_id':node,'display_name':name,'rank_rise':float(value),
                            'rank_2019':float(row[2019]),'rank_2025':float(row[2025])})
    cax=fig.add_axes([.14,.115,.72,.021])
    cb=fig.colorbar(sc,cax=cax,orientation='horizontal',ticks=RANK_BOUNDS,
                   spacing='uniform',drawedges=True)
    cb.ax.set_xticklabels(['−1,300','−600','−300','−100','−30','+30',
                          '+100','+300','+600','+1,300'])
    cb.outline.set_linewidth(.45)
    cb.ax.tick_params(length=3,width=.5,labelsize=14)
    cb.set_label('Change in full-SPRI rank (places; positive = rising risk priority)',
                 fontsize=15,labelpad=5)
    d.to_csv(SOURCE/'fig5d_city_risk_rank_change_2019_2025.csv')
    pd.DataFrame(annotations).to_csv(SOURCE/'fig5d_label_values.csv',index=False)
    save(fig,'fig5d_city_risk_rank_change_map')
    return {'ranked_cities':len(valid),
            'rank_rising_cities':int(d.rank_rise.gt(0).sum()),
            'rank_falling_cities':int(d.rank_rise.lt(0).sum()),
            'rank_unchanged_cities':int(d.rank_rise.eq(0).sum()),
            'rank_change_range':[float(d.rank_rise.min()),float(d.rank_rise.max())]}

if __name__=='__main__':
    figure4a()
    figure5c(shade=True)
    stats=figure5d()
    stats['risk_class_counts']=LATEST.quadrant.value_counts().to_dict()
    stats['source_sha256']={f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in
        ['SPRI_city_annual_2019_2025.csv','risk_method_manifest.json']}
    (SOURCE/'verification.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2),flush=True)
