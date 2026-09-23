"""
D1/D2 figures -- how BER and detectability behave for each jamming method, on the
3-D waveform link (README §3.3d/e). Reads the eval JSONs (no Sionna), so it runs
on the login node.

    python gan_figures.py

Three figures into artifacts/cgan/gan/run001/:
  fig_ber_vs_jsr.png     effectiveness: BER vs received JSR, every method
  fig_pdet_vs_jsr.png    detectability: P(det) vs JSR, one panel per detector, FAR dotted
  fig_frontier.png       the tradeoff: BER vs P(det) (parametric in JSR), per detector,
                         with the stealth budget (P(det) <= alpha) shaded

Compares: classical (noise, matched-QPSK/Amuru, genie ceiling) vs the reproduced
plain GAN (D1) vs the detector-aware trained GAN (D2) vs the shaped-noise control (D2a).
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GD="../artifacts/cgan/gan/run001"; BD="../artifacts/cgan/baselines/run001"; LD="../artifacts/cgan/learned/run001"
DETS=["power_one_sided","power_two_sided","kurtosis","spec_cnn"]
DET_LABEL={"power_one_sided":"power (1-sided)","power_two_sided":"power (2-sided)",
           "kurtosis":"kurtosis","spec_cnn":"spectrogram CNN"}
A="0.05"; ALPHA=0.05
BER_REF=1e-3     # the matched-BER operating point the headline is quoted at (README §3.3f)

def _at_matched_ber(j, det, target=BER_REF):
    """P(det) of `det` at the JSR where this jammer first reaches `target` BER."""
    jsr=j["jsr_db"]; ber=[p["ber"] for p in j["points"]]
    pdet=[p["pdet"].get(det,{}).get(A) for p in j["points"]]
    x=None
    for i in range(1,len(jsr)):
        b0,b1=ber[i-1],ber[i]
        if b0<target<=b1:
            if b0>0:
                l0,l1,lt=np.log(b0),np.log(b1),np.log(target)
                x=jsr[i-1]+(lt-l0)*(jsr[i]-jsr[i-1])/(l1-l0)
            else: x=jsr[i]
            break
    if x is None: return None
    for i in range(1,len(jsr)):
        if jsr[i-1]<=x<=jsr[i] and pdet[i-1] is not None and pdet[i] is not None:
            return pdet[i-1]+(x-jsr[i-1])*(pdet[i]-pdet[i-1])/(jsr[i]-jsr[i-1])
    return None

def pick_cnn_tag():
    """
    Which beta of the CNN-targeted D2 family to draw, by a stated rule rather than by
    hand: the generator with the LOWEST P(det) against the CNN at matched BER 1e-3 --
    the operating point the headline is quoted at (README §3.3f, user decision
    2026-09-23). The earlier rule ranked by confirmed stealthy BER at the alpha budget,
    where every row ties at 0, so the tie-break silently picked the LEAST interesting
    generator. Returns None if no spec_cnn eval exists, and the row is omitted.
    """
    best=None
    for f in sorted(glob.glob(f"{GD}/spec_cnn_b*.json")):
        j=json.load(open(f))
        if "points" not in j: continue
        v=_at_matched_ber(j,"spec_cnn")
        if v is None: continue
        if best is None or v<best[0]: best=(v, os.path.basename(f)[:-5])
    return best[1] if best else None
INK="#1f1f1e"; INK2="#5f5e5a"; GRID="#e4e3de"; BERFLOOR=3e-6

# Okabe-Ito (canonical CVD-safe categorical); color follows the method.
C={"noise":"#0072B2","optimal":"#E69F00","amuru":"#7a5195","genie":"#333333",
   "plain":"#56B4E9","eff":"#009E73","g_p1":"#D55E00","g_p2":"#c02a2a",
   "g_kurt":"#CC79A7","g_cnn":"#882255","s_p1":"#8C564B","s_kurt":"#8C564B","s_cnn":"#4d4d4d"}

def pts_series(jsr, points):
    return dict(jsr=np.array(jsr, float),
                ber=np.array([p["ber"] for p in points], float),
                pdet={d:np.array([p["pdet"].get(d,{}).get(A, np.nan) for p in points], float) for d in DETS})

def load():
    M=[]; cnn_tag=pick_cnn_tag()
    print(f"CNN-targeted D2 row drawn: {cnn_tag or '(none evaluated yet)'}"
          f"  (rule: lowest P(det)_CNN at matched BER {BER_REF:g})")
    d0=json.load(open(f"{BD}/sweep_K1.json")); j0=d0["jsr_db"]
    far={d: d0["clean"]["pdet"].get(d,{}).get(A) for d in DETS}
    def addp(key,label,jsr,points,dash=False,lw=1.8):
        s=pts_series(jsr,points); s.update(label=label,color=C[key],dash=dash,lw=lw); M.append(s)
    addp("noise","noise (barrage)",j0,d0["attacks"]["noise"])
    addp("optimal","matched QPSK / optimal (p=1)",j0,d0["attacks"]["pulsed_p1"])
    addp("amuru","Amuru pulsed (p=0.1)",j0,d0["attacks"]["pulsed_p0.1"])
    addp("genie","genie flip (ceiling)",j0,d0["attacks"]["omniscient_e1"],dash=True,lw=1.4)
    p1=json.load(open(f"{GD}/plain_run001.json")); addp("plain","plain GAN — D1 (run001)",p1["jsr_db"],p1["points"],lw=2.4)
    series=[("eff","eff","GAN eff, β=0 — D2"),
            ("g_p1","power_one_sided_b1000","GAN vs power, β=1000 — D2"),
            ("g_kurt","kurtosis_b100","GAN vs kurtosis, β=100 — D2")]
    if cnn_tag:
        series.append(("g_cnn",cnn_tag,f"GAN vs CNN, β={cnn_tag.split('_b')[-1]} — D2"))
    for key,tag,label in series:
        f=f"{GD}/{tag}.json"
        if os.path.exists(f):
            j=json.load(open(f)); addp(key,label,j["jsr_db"],j["points"],lw=2.4)
    # D2a shaped control (beta=1), per detector
    for key,target,label in [("s_kurt","kurtosis","shaped control vs kurtosis — D2a"),
                             ("s_cnn","spec_cnn","shaped control vs CNN — D2a")]:
        for f in glob.glob(f"{LD}/task*.json"):
            j=json.load(open(f)); t=j["result"].get("task",{})
            if isinstance(t,dict) and t.get("target")==target and t.get("beta")==1.0:
                runs=j["result"]["runs"]
                pts=[dict(ber=r["confirm"]["ber"],
                          pdet={d:{A:r["confirm"]["pdet"].get(d,{}).get(A)} for d in DETS}) for r in runs]
                addp(key,label,[r["jsr_db"] for r in runs],pts,dash=True,lw=1.6); break
    return M,far

def style(ax):
    ax.grid(True,color=GRID,lw=0.6); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)

def fig_ber(M):
    fig,ax=plt.subplots(figsize=(9,5.5))
    for s in M:
        ax.plot(s["jsr"], np.clip(s["ber"],BERFLOOR,None), color=s["color"],
                lw=s["lw"], ls="--" if s["dash"] else "-", label=s["label"])
    ax.set_yscale("log"); ax.set_ylim(BERFLOOR,0.6)
    ax.axhline(1e-3,color=INK2,lw=0.8,ls=":"); ax.text(ax.get_xlim()[0],1.1e-3,"BER 1e-3",color=INK2,fontsize=8,va="bottom")
    ax.set_xlabel("received JSR (dB)",color=INK); ax.set_ylabel("BER",color=INK)
    ax.set_title("Effectiveness — BER vs jammer power",loc="left",color=INK,fontsize=13)
    ax.legend(frameon=False,fontsize=8,ncol=2,loc="lower right"); style(ax)
    fig.tight_layout(); fig.savefig(f"{GD}/fig_ber_vs_jsr.png",dpi=140); plt.close(fig)

def fig_pdet(M,far):
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,det in zip(axes.flat,DETS):
        for s in M:
            ax.plot(s["jsr"], s["pdet"][det], color=s["color"], lw=s["lw"],
                    ls="--" if s["dash"] else "-", label=s["label"])
        ax.axhline(ALPHA,color="#c02a2a",lw=1.0,ls=":"); ax.text(ax.get_xlim()[0] if False else -49,ALPHA+0.02,f"FAR budget α={ALPHA}",color="#c02a2a",fontsize=8)
        ax.set_ylim(-0.03,1.03); ax.set_title(DET_LABEL[det],loc="left",color=INK,fontsize=11)
        ax.set_ylabel("P(detect)",color=INK); style(ax)
    for ax in axes[1]: ax.set_xlabel("received JSR (dB)",color=INK)
    axes[0,1].legend(frameon=False,fontsize=7.5,loc="center right")
    fig.suptitle("Detectability — P(detect) vs jammer power, per detector (FAR budget dotted)",
                 x=0.01,ha="left",color=INK,fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{GD}/fig_pdet_vs_jsr.png",dpi=140); plt.close(fig)

def fig_frontier(M,far):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,det in zip(axes.flat,DETS):
        ax.axvspan(0,ALPHA,color="#009E73",alpha=0.07)
        ax.axvline(ALPHA,color="#c02a2a",lw=1.0,ls=":")
        for s in M:
            x=s["pdet"][det]; y=np.clip(s["ber"],BERFLOOR,None)
            m=~np.isnan(x)
            ax.plot(x[m], y[m], color=s["color"], lw=s["lw"], ls="--" if s["dash"] else "-",
                    marker="o", ms=2.5, label=s["label"])
        ax.set_yscale("log"); ax.set_ylim(BERFLOOR,0.6); ax.set_xlim(-0.02,1.02)
        ax.set_title(DET_LABEL[det],loc="left",color=INK,fontsize=11)
        ax.set_ylabel("BER",color=INK); style(ax)
        ax.text(ALPHA/2,0.35,"stealth\nbudget",color="#2e7d32",fontsize=7.5,ha="center")
    for ax in axes[1]: ax.set_xlabel("P(detect) at this jammer power",color=INK)
    axes[0,1].legend(frameon=False,fontsize=7.5,loc="upper right")
    fig.suptitle("The tradeoff — BER vs P(detect). Stealthy-and-effective = top-left of the shaded band (empty)",
                 x=0.01,ha="left",color=INK,fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{GD}/fig_frontier.png",dpi=140); plt.close(fig)

def summary():
    """
    The headline number, straight from the eval JSONs: confirmed stealthy BER
    (max BER with P(det) <= alpha, re-measured on fresh frames) per jammer per
    detector. A row of zeros is the effectiveness/detectability wall (§3.3f);
    the D2 row that matters is the one trained against the column it sits in.
    """
    rows=[]
    d0=json.load(open(f"{BD}/sweep_K1.json"))
    for name in ["noise","pulsed_p1","pulsed_p0.1","omniscient_e1"]:
        rows.append((name, d0["confirmed"].get(name,{}), max(p["ber"] for p in d0["attacks"][name])))
    for f in sorted(glob.glob(f"{GD}/*.json")):
        tag=os.path.basename(f)[:-5]
        if tag.endswith("_smoke"): continue
        j=json.load(open(f))
        if "confirmed" not in j: continue
        rows.append((tag, j["confirmed"], max(p["ber"] for p in j["points"])))
    w=max(len(r[0]) for r in rows)
    print(f"\nConfirmed stealthy BER at alpha={ALPHA} (max BER with P(det) <= alpha, fresh frames)")
    print(f"{'jammer':<{w}} " + " ".join(f"{DET_LABEL[d]:>18}" for d in DETS) + f"{'max BER loud':>14}")
    for tag,conf,loud in rows:
        cells=[]
        for d in DETS:
            c=conf.get(d,{}).get(A)
            cells.append(f"{'--':>17} " if not c else
                         f"{c['confirmed_ber']:>17.2e}" + ("" if c.get("passed",True) else "!").ljust(1))
        print(f"{tag:<{w}} " + " ".join(cells) + f"{loud:>14.3f}")
    print("  '--' = no point met the budget;  '!' = the confirmation pass missed the budget")


if __name__=="__main__":
    plt.rcParams.update({"font.size":10,"axes.edgecolor":INK2,"text.color":INK,
                         "axes.labelcolor":INK,"figure.facecolor":"white","axes.facecolor":"white"})
    M,far=load()
    fig_ber(M); fig_pdet(M,far); fig_frontier(M,far)
    print("wrote fig_ber_vs_jsr.png, fig_pdet_vs_jsr.png, fig_frontier.png to", GD)
    for s in M: print(f"  {s['label']:34} max BER {np.nanmax(s['ber']):.3f}")
    summary()
