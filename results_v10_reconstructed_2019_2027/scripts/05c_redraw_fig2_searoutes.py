"""Redraw manuscript Fig. 2a-c with sea-network paths rather than geodesic arcs.

City colours retain the formal SO2/PM 2025-2020 and NOx 2025-2019
differences, with symmetric limits at the 98th percentile of absolute change.
Routes show 2025 health-weighted network flux, not pollutant change or AIS
tracks. searoute is used only for display geometry, never emission estimation.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
import searoute as sr
from searoute.searoute import setup_M, setup_P
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PC = ccrs.PlateCarree()
PROJ = ccrs.Robinson()
PANELS = [('SO2', 2020, 'fig3a1_so2_emission_network_change.png', (2830,1368), r'SO$_x$'),
          ('PM', 2020, 'fig3a2_pm_emission_network_change.png', (2807,1366), 'PM'),
          ('NOx',2019, 'fig3a3_nox_emission_network_change.png',(2830,1368), r'NO$_x$')]

def split_dateline(coords):
    """Split at +/-180 with interpolated boundary vertices, without a cross-map chord."""
    pts = np.asarray(coords, float)
    pts[:,0] = (pts[:,0]+180)%360-180
    parts, part = [], [pts[0].tolist()]
    for a,b in zip(pts[:-1],pts[1:]):
        if abs(b[0]-a[0]) <= 180:
            part.append(b.tolist())
            continue
        bx = b[0] + (-360 if b[0]>a[0] else 360)
        bound = -180 if bx<a[0] else 180
        lat = a[1]+(b[1]-a[1])*(bound-a[0])/(bx-a[0])
        part.append([bound,lat])
        parts.append(part)
        part = [[-bound,lat], b.tolist()]
    parts.append(part)
    return [p for p in parts if len(p)>1]

def routes(latest, edges, cache_file, top_n):
    coords=latest.set_index('node_id')[['rep_lon','rep_lat']]
    chosen=edges.nlargest(top_n,'w_t').copy()
    assert chosen[['From_City','To_City']].isin(coords.index).all().all()
    fingerprint=hashlib.sha256(chosen.to_csv(index=False).encode()+coords.to_csv().encode()).hexdigest()
    if cache_file.exists():
        saved=json.loads(cache_file.read_text(encoding='utf-8'))
        if saved.get('fingerprint')==fingerprint:
            print('Using verified sea-route cache',len(saved['routes']),flush=True)
            return saved
    M,P=setup_M(),setup_P()
    saved={'fingerprint':fingerprint,'year':2025,'top_n':top_n,
           'source':'network_edges_2025.parquet','geometry':'searoute maritime network',
           'append_orig_dest':False,'include_ports':False,'routes':[]}
    for i,r in enumerate(chosen.itertuples(index=False)):
        origin=coords.loc[r.From_City].to_numpy(float).tolist()
        dest=coords.loc[r.To_City].to_numpy(float).tolist()
        route=sr.searoute(origin,dest,M=M,P=P,append_orig_dest=False,include_ports=False)
        points=route['geometry']['coordinates']
        assert points and np.isfinite(points).all(),(r.From_City,r.To_City)
        saved['routes'].append({'from':r.From_City,'to':r.To_City,'w_t':r.w_t,
            'length_km':route['properties']['length'],
            'segments':split_dateline(points) if len(points)>1 else []})
        if (i+1)%100==0: print('Sea routes',i+1,'/',len(chosen),flush=True)
    cache_file.parent.mkdir(parents=True,exist_ok=True)
    cache_file.write_text(json.dumps(saved,ensure_ascii=False),encoding='utf-8')
    return saved

def draw_network(ax, route_data):
    weights=np.asarray([r['w_t'] for r in route_data['routes']])
    lo,hi=np.log1p(weights).min(),np.log1p(weights).max()
    segments,colors,widths=[],[],[]
    for r in sorted(route_data['routes'],key=lambda x:x['w_t']):
        q=(np.log1p(r['w_t'])-lo)/(hi-lo)
        for segment in r['segments']:
            assert max(abs(np.diff(np.asarray(segment)[:,0])),default=0)<=180.00001
            segments.append(segment)
            colors.append((.25,.46,.59,.025+.22*q**1.8))
            widths.append(.12+.72*q**1.6)
    # Draw below land so coastal simplification cannot create visible inland links.
    ax.add_collection(LineCollection(segments,colors=colors,linewidths=widths,
                      transform=PC,zorder=.4,capstyle='round',joinstyle='round'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=ROOT/'figures/established_panels/figure_3')
    ap.add_argument('--cache',type=Path,default=ROOT/'data/figure2_searoute_paths.json')
    ap.add_argument('--top-routes',type=int,default=1200)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    annual=pd.read_parquet(ROOT/'data/emission_city_annual_2019_2025.parquet')
    latest=annual.loc[annual.year.eq(2025)].copy()
    assert len(latest)==1425 and latest.node_id.is_unique
    edges=pd.read_parquet(ROOT/'data/network_edges_2025.parquet')
    sea=routes(latest,edges,args.cache,args.top_routes)
    # Recover the approved colour scale once, then retain it for reproducible redraws.
    if 'approved_colour_scale' not in sea:
        source=ROOT/'figures/established_panels/figure_3'/PANELS[0][2]
        rgba=np.asarray(Image.open(source).convert('RGB'))
        strip=rgba[:,round(rgba.shape[1]*.904),:]
        coloured=np.where(np.ptp(strip.astype(float),axis=1)>25)[0]
        lo,hi=int(coloured.min())+2,int(coloured.max())-2
        sea['approved_colour_scale']=(strip[lo:hi+1][::-1]/255.).tolist()
        args.cache.write_text(json.dumps(sea,ensure_ascii=False),encoding='utf-8')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,
        'mathtext.fontset':'stix','axes.labelcolor':'#202428','text.color':'#202428',
        'xtick.color':'#202428','ytick.color':'#202428','savefig.facecolor':'white'})
    cmap=LinearSegmentedColormap.from_list('formal_change',sea['approved_colour_scale'],N=256)
    audit={'routes':len(sea['routes']),'nonzero_geometry':sum(bool(r['segments']) for r in sea['routes']),
        'route_geometry':'searoute; snapped to marine nodes; no straight fallback',
        'data_changes':False,'panels':[]}
    for pol,base,filename,size,label in PANELS:
        wide=annual.pivot(index='node_id',columns='year',values=pol+'_t')
        delta=latest.node_id.map(wide[2025]-wide[base]).to_numpy()
        limit=float(np.quantile(np.abs(delta),.98))
        norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
        scale=np.clip(np.abs(delta)/limit,0,1)
        marker_size=3.5+29*scale**.5
        dpi=330
        fig=plt.figure(figsize=(size[0]/dpi,size[1]/dpi),dpi=dpi)
        ax=fig.add_axes([.012,.025,.855,.895],projection=PROJ)
        ax.set_global(); ax.set_facecolor('white')
        draw_network(ax,sea)
        ax.add_feature(cf.LAND.with_scale('50m'),facecolor='#F0F2F3',edgecolor='#AFBAC3',linewidth=.35,zorder=1)
        ax.add_feature(cf.BORDERS.with_scale('50m'),edgecolor='#C7D0D8',linewidth=.26,zorder=1.2)
        ax.spines['geo'].set_edgecolor('#A7B2BC'); ax.spines['geo'].set_linewidth(.65)
        order=np.argsort(np.abs(delta),kind='stable')
        sc=ax.scatter(latest.rep_lon.to_numpy()[order],latest.rep_lat.to_numpy()[order],
            c=delta[order],s=marker_size[order],norm=norm,cmap=cmap,transform=PC,
            edgecolors='white',linewidths=.22,alpha=.97,zorder=3)
        fig.text(.44,.962,f'Change in {label} ship emission, 2025 − {base}',ha='center',va='center',fontsize=11.5)
        cax=fig.add_axes([.898,.13,.017,.73])
        cb=fig.colorbar(sc,cax=cax); cb.ax.tick_params(labelsize=9,length=3,width=.65)
        cb.set_label(f'Change {base}→2025 (t; 98% colour limit)',fontsize=10.5,labelpad=8)
        cb.outline.set_linewidth(.65)
        handles=[Line2D([],[],color='#C8D8E4',lw=.6,label='Lower route emission'),
                 Line2D([],[],color='#7F9FB2',lw=1.4,label='Higher route emission')]
        legend=fig.legend(handles=handles,title='Shipping-route flux',loc='lower left',
            bbox_to_anchor=(.023,.032),frameon=True,facecolor='white',edgecolor='none',
            framealpha=.95,fontsize=9.3,title_fontsize=9.8,
            handlelength=2.1,labelspacing=.20,borderpad=.1)
        fig.savefig(args.output_dir/filename,dpi=dpi)
        plt.close(fig)
        audit['panels'].append({'file':filename,'pollutant':pol,'baseline_year':base,
            'latest_year':2025,'cities':len(delta),'colour_limit_t':limit,
            'delta_sum_t':float(delta.sum()),'delta_sha256':hashlib.sha256(delta.tobytes()).hexdigest(),
            'width':size[0],'height':size[1]})
        print('Rendered',filename,flush=True)
    (args.output_dir/'searoute_validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
