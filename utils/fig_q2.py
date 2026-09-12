import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figure_style as fs

def raw():
    l=pd.read_excel(fs.DATA/"附件2.xlsx",sheet_name="小区负载").iloc[:,1:].to_numpy(float)
    v=pd.read_excel(fs.DATA/"附件2.xlsx",sheet_name="光伏发电实际功率").iloc[:,1:].to_numpy(float)
    return l,v

def predictions(l,v):
    f1l=np.roll(l,7,axis=0); f2l=np.median(np.stack([np.roll(l,k,axis=0) for k in [7,14,21]]),axis=0)
    f1v=np.stack([np.roll(v,k,axis=0) for k in range(1,8)]).mean(0)
    return f1l-f1v,f2l-f1v,l-v

def diagnostics():
    l,v=raw(); f1,f2,a=predictions(l,v); sl=slice(31,None); rng=np.random.default_rng(2026); idx=rng.choice(a[sl].size,12000,replace=False)
    fig,axs=plt.subplots(1,2,figsize=(7.15,3.25))
    for pred,c,lab in [(f1[sl].ravel()[idx],fs.COLORS["blue"],"F1"),(f2[sl].ravel()[idx],fs.COLORS["orange"],"F2")]: axs[0].scatter(a[sl].ravel()[idx],pred,s=3,alpha=.10,color=c,label=lab,rasterized=True)
    lo,hi=np.percentile(a[sl],[1,99]); axs[0].plot([lo,hi],[lo,hi],ls="--",color=fs.COLORS["dark"]); axs[0].set(xlabel="实际净负荷/kW",ylabel="预测净负荷/kW"); axs[0].legend(); fs.clean(axs[0])
    e=[(f1-a)[sl].ravel(),(f2-a)[sl].ravel()]; axs[1].boxplot(e,positions=[1,2],showfliers=False,widths=.45,patch_artist=True,boxprops={"facecolor":"#D9EAF4"},medianprops={"color":fs.COLORS["red"]}); axs[1].set_xticks([1,2],["F1","F2"]); axs[1].set_ylabel("净负荷残差/kW",fontproperties=fs.CN); axs[1].axhline(0,color=fs.COLORS["gray"],lw=.8); fs.clean(axs[1])
    fs.panel_name(axs[0],"（a）预测值与实际值"); fs.panel_name(axs[1],"（b）净负荷残差分布"); fig.subplots_adjust(bottom=.23,wspace=.30); fs.save(fig,"q2-diagnostics")

def safe_curve():
    l,v=raw(); f1,_,a=predictions(l,v); day=(pd.Timestamp("2025-03-20")-pd.Timestamp("2025-01-01")).days; hist=(f1-a)[day-35:day].ravel(); margin=np.quantile(hist,.805); t=np.arange(144)/6
    fig,ax=plt.subplots(figsize=(7.15,3.2)); ax.plot(t,a[day],color=fs.COLORS["dark"],label="实际净负荷"); ax.plot(t,f1[day],color=fs.COLORS["blue"],label="F1基础预测"); ax.plot(t,f1[day]+margin,color=fs.COLORS["orange"],label="风险修正预测"); ax.fill_between(t,f1[day],f1[day]+margin,color=fs.COLORS["orange"],alpha=.18,label="风险裕量"); ax.set_ylabel("净负荷/kW",fontproperties=fs.CN); fs.time_axis(ax); fs.clean(ax); ax.legend(ncol=4,loc="upper center"); fs.save(fig,"q2-safe")

def days():
    z=np.load(fs.RESULTS/"q2_bundle.npz"); dates=["2025-03-20","2025-06-21","2025-09-23","2025-12-21"]; ids=[(pd.Timestamp(d)-pd.Timestamp("2025-01-01")).days for d in dates]; names=["春分","夏至","秋分","冬至"]; t=np.arange(144)/6
    fig,axs=plt.subplots(2,4,figsize=(7.25,4.15),sharex=True,sharey="row")
    for j,(i,n) in enumerate(zip(ids,names)):
        axs[0,j].step(t,z["P"][i]*6,where="post",color=fs.COLORS["blue"]); axs[0,j].fill_between(t,0,z["U"][i]*6,step="post",color=fs.COLORS["red"],alpha=.38); fs.clean(axs[0,j]); axs[0,j].text(.5,1.03,n,transform=axs[0,j].transAxes,ha="center",fontproperties=fs.CN)
        axs[1,j].plot(t,z["S"][i,1:] if z["S"].shape[1]==145 else z["S"][i],color=fs.COLORS["purple"]); axs[1,j].axhline(1200,ls="--",lw=.6,color=fs.COLORS["gray"]); axs[1,j].axhline(10800,ls="--",lw=.6,color=fs.COLORS["gray"]); fs.clean(axs[1,j]); axs[1,j].set_xticks([0,12,24])
    axs[0,0].set_ylabel("购电功率/kW",fontproperties=fs.CN); axs[1,0].set_ylabel("储电量/kWh",fontproperties=fs.CN); fig.supxlabel("时刻/h",fontproperties=fs.CN,y=.02); fig.subplots_adjust(hspace=.13,wspace=.12,bottom=.12); fs.save(fig,"q2-days")

def sensitivity():
    d=pd.read_csv(fs.RESULTS/"问题2_配置敏感性.csv"); base=d.iloc[0]; d=d.iloc[:min(12,len(d))].copy(); labels=d["方案"].astype(str).str.replace("主配置","基准",regex=False); delta=(d["cash_total"]-base["cash_total"])/10000
    order=np.argsort(delta); y=np.arange(len(d)); fig,axs=plt.subplots(1,2,figsize=(7.15,3.75),sharey=True)
    axs[0].hlines(y,0,delta.iloc[order],color="#B8C6CF"); axs[0].scatter(delta.iloc[order],y,c=np.where(delta.iloc[order]<=0,fs.COLORS["green"],fs.COLORS["red"]),s=28); axs[0].axvline(0,color=fs.COLORS["dark"],lw=.8); axs[0].set_yticks(y,labels.iloc[order],fontproperties=fs.CN); axs[0].set_xlabel("相对基准费用变化/万元",fontproperties=fs.CN); fs.clean(axs[0])
    em=(d["emergency_kwh"]-base["emergency_kwh"])/10000; axs[1].hlines(y,0,em.iloc[order],color="#B8C6CF"); axs[1].scatter(em.iloc[order],y,color=fs.COLORS["orange"],s=28); axs[1].axvline(0,color=fs.COLORS["dark"],lw=.8); axs[1].set_xlabel("应急购电变化/万kWh",fontproperties=fs.CN); fs.clean(axs[1]); fs.panel_name(axs[0],"（a）年度费用"); fs.panel_name(axs[1],"（b）应急购电量"); fig.subplots_adjust(bottom=.22,wspace=.12,left=.27); fs.save(fig,"q2-sensitivity")

def main(): fs.setup(); diagnostics(); safe_curve(); days(); sensitivity()
if __name__=="__main__": main()
