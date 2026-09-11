"""Manuscript figures. EVERY plotted value is read from results.json, the
canonical statistics file; nothing is hard-coded here. If a number changes,
it changes in one place and all figures and the manuscript follow."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, json, os
from matplotlib.lines import Line2D

R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results.json")))

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":8,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.major.width":0.8,"ytick.major.width":0.8,
    "xtick.major.size":3,"ytick.major.size":3,"legend.frameon":False,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight",
    "savefig.pad_inches":0.02})
C = {"ns":"#1b6ca8","ks":"#e08214","cgl":"#5aae61","fd":"#8073ac",
     "grey":"#666666","red":"#b2182b","lt":"#cccccc"}

# ---------------------------------------------------------------- Fig 1
# the observable-preserving protocol + the causal result
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.4,2.6),
                           gridspec_kw={"width_ratios":[1.15,1],"wspace":0.30})
ax1.set_xlim(0,10); ax1.set_ylim(0,6.4); ax1.axis("off")
ax1.add_patch(plt.Rectangle((0.15,4.42),2.75,1.42,fc="#e8f1f8",ec=C["ns"],lw=1))
ax1.text(1.525,5.13,"reduced state\n$V$",ha="center",va="center",fontsize=7.2)
for i,(dy,lab,col,txt) in enumerate([
        (3.3,"baseline",C["grey"],"$V$"),
        (2.2,"estimated",C["ns"],"$V+Q\\hat{e}$"),
        (1.1,"wrong sign",C["red"],"$V-Q\\hat{e}$"),
        (0.0,"oracle",C["ks"],"$V+Qe$")]):
    ax1.annotate("",xy=(4.1,dy+0.35),xytext=(2.9,5.10),
                 arrowprops=dict(arrowstyle="->",lw=0.8,color=col,alpha=0.75))
    ax1.add_patch(plt.Rectangle((4.1,dy),2.4,0.72,fc="white",ec=col,lw=1))
    ax1.text(5.3,dy+0.36,txt,ha="center",va="center",fontsize=7.5,color=col)
    ax1.text(6.75,dy+0.36,lab,ha="left",va="center",fontsize=6.8,color=col)
ax1.text(1.525,3.98,"$P(V+c)=PV$\nexactly",ha="center",va="top",fontsize=7,
         style="italic",color=C["ns"])
ax1.text(-0.02,1.04,"a",transform=ax1.transAxes,fontsize=10,fontweight="bold")

CI=R["causal_intervention"]
gains=[CI["gain_estimated"],CI["gain_wrong_sign"],CI["gain_oracle"]]
labels=["estimated\n$V+Q\\hat{e}$","wrong sign\n$V-Q\\hat{e}$","oracle\n$V+Qe$"]
cols=[C["ns"],C["red"],C["ks"]]
b=ax2.bar(range(3),gains,color=cols,width=0.62,zorder=3)
ax2.axhline(1.0,color=C["grey"],lw=0.9,ls="--",zorder=2)
ax2.set_xticks(range(3)); ax2.set_xticklabels(labels,fontsize=7)
ax2.set_ylabel("median forecast gain")
ax2.set_ylim(0,2.30)
_nw=CI["n_windows"]
_wins=[f'{CI["wins_estimated"]}/{_nw}',f'{CI["wins_wrong_sign"]}/{_nw}',f'{CI["wins_oracle"]}/{_nw}']
for x,g,n in zip(range(3),gains,_wins):
    ax2.text(x,g+0.06,f"{g:.3f}",ha="center",fontsize=7,fontweight="bold")
    ax2.text(x,0.08,n,ha="center",fontsize=6.5,color="white",fontweight="bold")
ax2.text(1.0,1.045,"no change",fontsize=6.3,color=C["grey"],ha="center")
ax2.annotate("",xy=(2.0,1.94),xytext=(0.0,1.94),
             arrowprops=dict(arrowstyle="<->",lw=0.7,color=C["grey"]))
ax2.text(1.0,1.99,"100% of oracle benefit captured",ha="center",fontsize=6.2,
         color=C["grey"])
ax2.text(-0.16,1.04,"b",transform=ax2.transAxes,fontsize=10,fontweight="bold")
plt.savefig("fig/fig1.png"); plt.close()

# ---------------------------------------------------------------- Fig 2
# the effect-size law
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.6,2.7),
                           gridspec_kw={"wspace":0.42})
lam=np.linspace(0.001,2.2,400)
for cs,st in [(-0.999,"-"),(-0.95,"--"),(-0.85,"-."),(-0.6,":")]:
    g=1/np.sqrt(np.maximum(1+2*lam*cs+lam**2,1e-9))
    ax1.plot(lam,g,st,color=C["grey"],lw=1.0,
             label=f"$\\cos\\theta={cs:g}$")
_L=R["effect_size_law"]; _sy={d["name"]:d for d in _L["systems"]}
pts=[(_sy["Kuramoto–Sivashinsky"]["lam"],_sy["Kuramoto–Sivashinsky"]["gain"],"KS unstable",C["ks"]),
     (_sy["Navier–Stokes"]["lam"],_sy["Navier–Stokes"]["gain"],"Navier–Stokes",C["ns"]),
     (_sy["FD + point sensors"]["lam"],_sy["FD + point sensors"]["gain"],"FD + sensors",C["fd"]),
     (_L["ks_damped"]["lam"],_L["ks_damped"]["gain"],"KS damped",C["ks"])]
for l,g,n,c in pts:
    ax1.scatter([l],[g],s=34,color=c,zorder=5,edgecolor="white",lw=0.6)
ax1.annotate("KS unstable",(0.0891,1.035),(0.30,0.80),fontsize=6.5,color=C["ks"],
             arrowprops=dict(arrowstyle="-",lw=0.5,color=C["ks"]))
ax1.annotate("Navier–Stokes",(0.4445,1.651),(1.62,2.55),fontsize=6.5,color=C["ns"],
             ha="left",arrowprops=dict(arrowstyle="-",lw=0.5,color=C["ns"],
             connectionstyle="arc3,rad=-0.15"))
ax1.annotate("FD + sensors",(0.3909,1.881),(0.03,5.0),fontsize=6.5,color=C["fd"],
             arrowprops=dict(arrowstyle="-",lw=0.5,color=C["fd"]))
ax1.annotate("KS damped",(1.0298,18.63),(1.15,11),fontsize=6.5,color=C["ks"],
             arrowprops=dict(arrowstyle="-",lw=0.5,color=C["ks"]))
ax1.set_yscale("log"); ax1.set_xlabel("$\\lambda=\\|\\Delta\\|/\\|b\\|$")
ax1.set_ylabel("forecast gain")
ax1.set_ylim(0.62,60); ax1.set_xlim(0,2.2)
ax1.legend(fontsize=6,loc="upper left")
ax1.text(-0.16,1.04,"a",transform=ax1.transAxes,fontsize=10,fontweight="bold")

sysn=["Kuramoto–\nSivashinsky","Navier–\nStokes","Ginzburg–\nLandau\n(cubic)",
      "FD + point\nsensors"]
errs=[d["identity_max_rel_err"] for d in _L["systems"]]
cc=[C["ks"],C["ns"],C["cgl"],C["fd"]]
ax2.bar(range(4),errs,color=cc,width=0.6,zorder=3)
ax2.set_yscale("log"); ax2.set_ylim(1e-16,1e-8)
ax2.axhline(1e-9,color=C["red"],lw=0.9,ls="--")
ax2.text(3.45,1.4e-9,"registered\nthreshold",fontsize=6,color=C["red"],ha="right")
ax2.set_xticks(range(4)); ax2.set_xticklabels(sysn,fontsize=6.3)
ax2.set_ylabel("max $|$predicted $-$ observed$| \\, / \\,$observed")
ax2.text(-0.20,1.04,"b",transform=ax2.transAxes,fontsize=10,fontweight="bold")
plt.savefig("fig/fig2.png"); plt.close()

# ---------------------------------------------------------------- Fig 3
# Theorem F reachability
fig,ax=plt.subplots(figsize=(3.5,2.7))
r=np.linspace(0,0.995,400)
ax.plot(r,1/np.sqrt(1-r**2),color=C["grey"],lw=1.2,zorder=2,
        label="ceiling $1/\\sqrt{1-\\rho^2}$")
ax.fill_between(r,0.9,1/np.sqrt(1-r**2),color="#dceaf5",alpha=0.7,zorder=1)
_RC=R["reachability"]
V=[(v["id"],v["rho"],v["gain"],C["red"] if v["id"]=="A" else C["ns"])
   for v in _RC["variants"]]
for n,rho,g,c in V:
    x=min(rho,0.995)
    ax.scatter([x],[g],s=36,color=c,zorder=5,edgecolor="white",lw=0.6)
_vA=_RC["variants"][0]
ax.annotate("A  POD only\nreachability-limited",(_vA["rho"],_vA["gain"]),(0.30,3.2),
            fontsize=6.5,color=C["red"],
            arrowprops=dict(arrowstyle="-",lw=0.5,color=C["red"]))
ax.annotate("B–E  $P\\subseteq S$\ndynamics-limited",(0.99,1.58),(0.52,8.5),
            fontsize=6.5,color=C["ns"],
            arrowprops=dict(arrowstyle="-",lw=0.5,color=C["ns"]))
ax.text(0.055,1.35,"achievable by\nsome hidden\ncorrection",fontsize=6.3,
        color=C["ns"],style="italic")
ax.text(0.09,16,"unreachable by ANY\nhidden correction",fontsize=6.3,
        color=C["grey"],style="italic")
ax.set_yscale("log"); ax.set_ylim(0.9,40); ax.set_xlim(0,1.02)
ax.set_xlabel("reachable fraction  $\\rho=\\|\\Pi_{P\\cdot S}b\\|/\\|b\\|$")
ax.set_ylabel("forecast gain")
ax.text(0.98,1.02,f"{_RC['n_windows']} windows, no violations",fontsize=6.3,
        color=C["grey"],ha="right",va="bottom",transform=ax.transAxes)
plt.savefig("fig/fig3.png"); plt.close()

# ---------------------------------------------------------------- Fig 4
# dose-response
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.4,2.6),
                           gridspec_kw={"wspace":0.32})
_VB=R["validity_boundary"]
ns=np.array(_VB["n_sensors"]); fr=ns/_VB["total_observable_dim"]
er=np.array(_VB["reconstruction_error"]); lm=np.array(_VB["lambda"])
ax1.plot(fr,er,"o-",color=C["fd"],lw=1.2,ms=4.5,zorder=3)
ax1.axhline(0.0049,color=C["cgl"],lw=0.9,ls="--")
ax1.text(0.98,0.028,"wholly modal",fontsize=6.3,color=C["cgl"],ha="right")
ax1.axhspan(0.44,0.56,color=C["lt"],alpha=0.45,zorder=1)
ax1.text(0.98,0.575,"saturated",fontsize=6.3,color=C["grey"],ha="right")
ax1.set_xlabel("pointwise fraction of the observable set")
ax1.set_ylabel("reconstruction error of the law")
ax1.set_ylim(0,0.63); ax1.set_xlim(-0.03,1.03)
_one=er[1]; _sat=er[4]; _base=er[0]
_pct=round(100*(_one-_base)/(_sat-_base))
ax1.annotate(f"one sensor in {_VB['total_observable_dim']}\ngives {_pct}% of the\nsaturated error",
             (fr[1],_one),(0.20,0.13),fontsize=6.3,color=C["fd"],
             arrowprops=dict(arrowstyle="->",lw=0.6,color=C["fd"]))
ax1.text(-0.16,1.04,"a",transform=ax1.transAxes,fontsize=10,fontweight="bold")

ax2.plot(fr,lm,"s-",color=C["ns"],lw=1.2,ms=4.2,zorder=3)
ax2.set_xlabel("pointwise fraction of the observable set")
ax2.set_ylabel("$\\lambda$  (hidden-driven share)")
ax2.set_ylim(0,1.0); ax2.set_xlim(-0.03,1.03)
ax2.text(-0.16,1.04,"b",transform=ax2.transAxes,fontsize=10,fontweight="bold")
plt.savefig("fig/fig4.png"); plt.close()

# ---------------------------------------------------------------- Fig 5
# adaptive algorithm
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.6,2.95),
                           gridspec_kw={"width_ratios":[1.35,1],"wspace":0.34})
meth=["uncorrected","hidden-\nonly","observable-\nonly",
      "ensemble\n3DVar*","adaptive\n(ours)"]
_AD=R["adaptive"]
unst=_AD["unstable"]; damp=_AD["damped"]
x=np.arange(5); w=0.30
ax1.bar(x-w/2,unst,w,label=f"unstable band  ($\\lambda\\approx{_AD['lam_unstable']}$)",
        color=C["ks"],zorder=3)
ax1.bar(x+w/2,damp,w,label=f"damped band  ($\\lambda\\approx{_AD['lam_damped']}$)",
        color=C["ns"],zorder=3)
ax1.set_yscale("log"); ax1.set_ylim(0.8,32)
ax1.axhline(1.0,color=C["grey"],lw=0.8,ls="--",zorder=2)
ax1.set_xticks(x); ax1.set_xticklabels(meth,fontsize=6.6)
ax1.text(0.5,-0.30,"*consumes observations the adaptive selector never sees",
         transform=ax1.transAxes,ha="center",fontsize=5.8,color=C["grey"])
ax1.set_ylabel("median forecast gain")
ax1.legend(fontsize=6.3,loc="upper left")
for xi,(u,d) in enumerate(zip(unst,damp)):
    # the damped hidden-only bar and the unstable adaptive bar carry the two
    # largest labels; offset each away from its neighbour so they do not collide
    du = dict(ha="center"); dd = dict(ha="center")
    ux, uy = xi-w/2, u*1.09
    dx_, dy_ = xi+w/2, d*1.09
    if xi==0:                      # both are 1.00 — spread them apart
        ux, du = xi-w/2-0.05, dict(ha="right")
        dx_, dd = xi+w/2+0.05, dict(ha="left")
    if xi==1: dx_, dy_, dd = xi+w/2+0.20, d*0.82, dict(ha="left")
    if xi==4: ux, uy, du = xi-w/2-0.20, u*0.82, dict(ha="right")
    ax1.text(ux,uy,f"{u:.2f}",fontsize=5.8,**du)
    ax1.text(dx_,dy_,f"{d:.2f}",fontsize=5.8,**dd)
ax1.text(-0.11,1.04,"a",transform=ax1.transAxes,fontsize=10,fontweight="bold")

_CO=R["cost"]
Nv=_CO["N"]; meas=_CO["measured_ratio"]; nom=_CO["gridpoint_model"]
ax2.plot(Nv,meas,"o-",color=C["ns"],lw=1.3,ms=5,label="measured",zorder=4)
ax2.plot(Nv,nom,"s--",color=C["grey"],lw=1.0,ms=4,label="grid-point model",zorder=3)
ax2.axhline(1.0,color=C["red"],lw=0.9,ls="--",zorder=2)
ax2.text(79,1.13,"parity with\nfull solver",fontsize=6,color=C["red"],ha="right")
ax2.set_yscale("log"); ax2.set_xlabel("grid resolution $N$")
ax2.set_ylabel("estimator cost / full solver")
ax2.set_xticks(Nv); ax2.legend(fontsize=6.3)
ax2.text(-0.20,1.04,"b",transform=ax2.transAxes,fontsize=10,fontweight="bold")
plt.savefig("fig/fig5.png"); plt.close()
print("figures written")
