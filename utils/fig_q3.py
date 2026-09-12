import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figure_style as fs

def forecast_quality():
    d=pd.read_csv(fs.RESULTS/"问题3_预见期误差_分来源.csv"); d=d[d["发布时刻_h"].isin([6,12,18])]
    fig,axs=plt.subplots(1,2,figsize=(7.15,3.25)); colors=[fs.COLORS["blue"],fs.COLORS["orange"],fs.COLORS["purple"]]
    for (r,g),c in zip(d.groupby("发布时刻_h"),colors):
        axs[0].plot(g["提前期_h"],g["MAE_kW"],marker="o",ms=3,color=c,label=f"{r}:00")
        axs[1].plot(g["提前期_h"],g["尺度_s_kW"],marker="o",ms=3,color=c,label=f"{r}:00")
    for ax in axs: ax.set_xlabel("提前期/h",fontproperties=fs.CN); ax.set_xticks([1,6,12,18,24]); ax.legend(ncol=3); fs.clean(ax)
    axs[0].set_ylabel("MAE/kW"); axs[1].set_ylabel("风险尺度 $s_h$/kW",fontproperties=fs.CN)
    fs.panel_name(axs[0],"（a）校正预报误差"); fs.panel_name(axs[1],"（b）分层风险尺度"); fig.subplots_adjust(bottom=.23,wspace=.28); fs.save(fig,"q3-forecast-quality")

def contract():
    z=np.load(fs.RESULTS/"q3_bundle.npz"); i=(pd.Timestamp("2025-03-20")-pd.Timestamp("2025-01-01")).days; t=np.arange(144)/6; p=z["main_P"][i]*6; q=z["main_Q"][i]*6; s=z["main_S"][i]; s=s[1:] if len(s)==145 else s
    fig,axs=plt.subplots(2,1,figsize=(7.15,4.2),sharex=True,gridspec_kw={"height_ratios":[1.5,1]})
    ax=axs[0]; ax.step(t,p,where="post",color=fs.COLORS["gray"],label="0:00原计划 p(t)"); ax.step(t,q,where="post",color=fs.COLORS["blue"],label="最终合同 q(t)"); ax.fill_between(t,p,q,where=q>=p,step="post",color=fs.COLORS["red"],alpha=.28,label="上调"); ax.fill_between(t,p,q,where=q<p,step="post",color=fs.COLORS["green"],alpha=.28,label="下调"); ax.set_ylabel("购电功率/kW",fontproperties=fs.CN); ax.legend(ncol=4); fs.clean(ax)
    ax=axs[1]; ax.plot(t,s,color=fs.COLORS["purple"]); ax.fill_between(t,1200,s,color=fs.COLORS["purple"],alpha=.12); ax.set_ylabel("储电量/kWh",fontproperties=fs.CN); fs.time_axis(ax); fs.clean(ax)
    for ax in axs:
        for h in [6,12,18]: ax.axvline(h,color=fs.COLORS["orange"],ls="--",lw=.8)
    fs.save(fig,"q3-contract")

def ablation():
    d=pd.read_csv(fs.RESULTS/"问题3_信息消融.csv"); order=["{}","{6}","{12}","{18}","{6,12}","{6,18}","{12,18}","{6,12,18}"]; d=d.set_index("获准集合S").reindex(order).reset_index(); labels=["空集","{6}","{12}","{18}","{6,12}","{6,18}","{12,18}","{6,12,18}"]; y=np.arange(8); fig,axs=plt.subplots(1,2,figsize=(7.15,3.55),sharey=True)
    c=[fs.COLORS["gray"]]+[fs.COLORS["blue"]]*6+[fs.COLORS["red"]]
    axs[0].scatter(d["现金总费_元"]/10000,y,c=c,s=38,zorder=3); axs[0].plot(d["现金总费_元"]/10000,y,color="#BEC7CD",lw=.8); axs[0].set_xlabel("年度费用/万元",fontproperties=fs.CN); axs[0].set_yticks(y,labels); axs[0].invert_yaxis(); fs.clean(axs[0])
    axs[1].scatter(d["应急购电量_kWh"]/10000,y,c=c,s=38,zorder=3); axs[1].plot(d["应急购电量_kWh"]/10000,y,color="#BEC7CD",lw=.8); axs[1].set_xlabel("应急购电量/万kWh",fontproperties=fs.CN); fs.clean(axs[1])
    fs.panel_name(axs[0],"（a）年度现金费用"); fs.panel_name(axs[1],"（b）应急购电量"); fig.subplots_adjust(bottom=.22,wspace=.12,left=.16); fs.save(fig,"q3-ablation")

def extra_time():
    d=pd.read_csv(fs.RESULTS/"问题3_其他时刻单次边际测试.csv"); x=d["候选时刻_h"].to_numpy(); y=d["kappa_star_元每次"].to_numpy(); pos=d["节省为正天数"].to_numpy()/d["验证天数"].to_numpy()*100
    fig,ax=plt.subplots(figsize=(7.15,3.45)); ax.axhline(0,color=fs.COLORS["dark"],lw=.8); colors=[fs.COLORS["red"] if v<0 else fs.COLORS["green"] for v in y]
    ax.vlines(x,0,y,color=colors,lw=2); ax.scatter(x,y,c=colors,s=55,zorder=3)
    for xx,yy,pp in zip(x,y,pos): ax.annotate(f"{yy:.0f} 元/次\n正收益日 {pp:.0f}%",(xx,yy),xytext=(0,8 if yy>=0 else -10),textcoords="offset points",ha="center",va="bottom" if yy>=0 else "top",fontproperties=fs.CN,fontsize=8)
    if 21 in x:
        yy=y[np.where(x==21)[0][0]]; ax.scatter([21],[yy],s=110,facecolors="white",edgecolors=fs.COLORS["dark"],lw=1.4,zorder=4); ax.scatter([21],[yy],marker="x",s=55,color=fs.COLORS["dark"],zorder=5); ax.axvspan(20.3,21.7,color="#E6E6E6",alpha=.65)
        ax.text(21,ax.get_ylim()[0]*.76,"21:00后光伏为0\n数值变化来自合同重优化\n不建议设置光伏预报",ha="center",va="bottom",fontproperties=fs.CN,fontsize=8,color=fs.COLORS["dark"])
    ax.set_xticks(x,[f"{i}:00" for i in x]); ax.set_xlabel("候选更新时间",fontproperties=fs.CN); ax.set_ylabel(r"盈亏平衡成本 $\kappa_r^*$/（元/次）",fontproperties=fs.CN); fs.clean(ax); fs.save(fig,"q3-extra-time")

def main(): fs.setup(); forecast_quality(); contract(); ablation()
if __name__=="__main__": main()
