import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figure_style as fs

def price_predictions():
    p=pd.read_excel(fs.DATA/"附件4.xlsx").iloc[:,1:].to_numpy(float); preds={}
    preds["昨日同槽"]=np.roll(p,1,axis=0); preds["近7日均值"]=np.stack([np.roll(p,k,axis=0) for k in range(1,8)]).mean(0); preds["近30日均值"]=np.stack([np.roll(p,k,axis=0) for k in range(1,31)]).mean(0); preds["近30日中位数"]=np.median(np.stack([np.roll(p,k,axis=0) for k in range(1,31)]),axis=0); return p,preds

def price_diagnostics():
    p,preds=price_predictions(); a=p[31:].ravel(); m=preds["近30日中位数"][31:].ravel(); fig,axs=plt.subplots(1,2,figsize=(7.15,3.25)); hb=axs[0].hexbin(a,m,gridsize=40,mincnt=1,cmap="Blues",bins="log"); lo,hi=np.percentile(a,[.5,99.5]); axs[0].plot([lo,hi],[lo,hi],ls="--",color=fs.COLORS["red"]); axs[0].set(xlabel="实际电价/(元/kWh)",ylabel="预测电价/(元/kWh)"); fs.clean(axs[0]); fig.colorbar(hb,ax=axs[0],fraction=.05,pad=.02).set_label("对数频数",fontproperties=fs.CN)
    errors=[np.abs(v[31:]-p[31:]).ravel() for v in preds.values()]; axs[1].boxplot(errors,showfliers=False,patch_artist=True,boxprops={"facecolor":"#D9EAF4"},medianprops={"color":fs.COLORS["red"]}); axs[1].set_xticks(range(1,5),["昨日","7日均值","30日均值","30日中位数"],rotation=18,fontproperties=fs.CN); axs[1].set_ylabel("绝对误差/(元/kWh)",fontproperties=fs.CN); fs.clean(axs[1]); fs.panel_name(axs[0],"（a）预测与实际电价"); fs.panel_name(axs[1],"（b）基准方法误差分布"); fig.subplots_adjust(bottom=.25,wspace=.33); fs.save(fig,"q4-price-diagnostics")

def outcome():
    d=pd.read_csv(fs.RESULTS/"问题4_方案对比.csv"); f=d[d["tag"].str.contains("median30") & ~d["tag"].str.contains("center")]; # retain principal modes
    vals={r["tag"]:r["cash_total"] for _,r in d.iterrows()}; fig,axs=plt.subplots(1,2,figsize=(7.15,3.35))
    q2=pd.read_csv(fs.RESULTS/"全题关键指标.csv"); fixed=float(q2.loc[q2["问题"]=="Q2","现金费用_元"].iloc[0]); q3=float(q2.loc[q2["问题"]=="Q3","现金费用_元"].iloc[0]); q42=vals.get("Q4-2_median30",17378687.69); q43=vals.get("Q4-3_median30",15465962.60)
    for j,(name,a,b) in enumerate([("固定电价",fixed,q3),("波动电价",q42,q43)]): axs[0].plot([a/1e4,b/1e4],[j,j],color="#AAB7C0",lw=2); axs[0].scatter([a/1e4],[j],color=fs.COLORS["gray"],s=45); axs[0].scatter([b/1e4],[j],color=fs.COLORS["blue"],s=45); axs[0].text((a+b)/2/1e4,j+.14,f"节省 {(a-b)/1e4:.1f} 万元",ha="center",fontproperties=fs.CN,fontsize=8)
    axs[0].set_yticks([0,1],["固定电价","波动电价"],fontproperties=fs.CN); axs[0].set_xlabel("年度费用/万元",fontproperties=fs.CN); fs.clean(axs[0])
    oracle42=vals.get("Q4-2_oracle",q42-78602.31); oracle43=vals.get("Q4-3_oracle",q43-82912.80)
    for j,(a,b) in enumerate([(oracle42,q42),(oracle43,q43)]): axs[1].plot([a/1e4,b/1e4],[j,j],color="#AAB7C0",lw=2); axs[1].scatter(a/1e4,j,color=fs.COLORS["green"],s=45); axs[1].scatter(b/1e4,j,color=fs.COLORS["orange"],s=45); axs[1].text((a+b)/2/1e4,j+.14,f"差 {(b-a)/1e4:.1f} 万元",ha="center",fontproperties=fs.CN,fontsize=8)
    axs[1].set_yticks([0,1],["固定合同","可调合同"],fontproperties=fs.CN); axs[1].set_xlabel("年度费用/万元",fontproperties=fs.CN); fs.clean(axs[1]); fs.panel_name(axs[0],"（a）合同调整权的价值"); fs.panel_name(axs[1],"（b）未知价格的代价"); fig.subplots_adjust(bottom=.23,wspace=.38,left=.13); fs.save(fig,"q4-outcome")

def risk_stability():
    files=[("Q2", "问题2_逐日汇总.csv"),("Q3","问题3_逐日汇总.csv"),("Q4-2","问题4-2_逐日汇总.csv"),("Q4-3","问题4-3_逐日汇总.csv")]; data=[pd.read_csv(fs.RESULTS/f)["现金费_元"].to_numpy()/10000 for _,f in files]; fig,axs=plt.subplots(1,2,figsize=(7.15,3.45)); bp=axs[0].boxplot(data,tick_labels=[x[0] for x in files],showfliers=False,patch_artist=True); [b.set_facecolor("#D9EAF4") for b in bp["boxes"]]; axs[0].set_ylabel("日费用/万元",fontproperties=fs.CN); fs.clean(axs[0])
    m=pd.read_csv(fs.RESULTS/"尾部风险_月度配对差.csv"); m=m[(m["方案A"].str.startswith("Q3")) & (m["方案B"].str.contains("空集"))]; axs[1].axhline(0,color=fs.COLORS["dark"],lw=.8); axs[1].plot(np.arange(len(m)),m["A相对B日均差_元"],marker="o",color=fs.COLORS["green"]); axs[1].set_xticks(np.arange(len(m)),m["月份"].str[-2:].astype(int)); axs[1].set_xlabel("月份",fontproperties=fs.CN); axs[1].set_ylabel("Q3相对无更新日均费用差/元",fontproperties=fs.CN); fs.clean(axs[1]); fs.panel_name(axs[0],"（a）主方案日费用分布"); fs.panel_name(axs[1],"（b）信息价值的跨月稳定性"); fig.subplots_adjust(bottom=.23,wspace=.30); fs.save(fig,"risk-stability")

def sensitivity_overview():
    d=pd.read_csv(fs.RESULTS/"模型评价_结构敏感性.csv"); d=d[(d["group"].isin(["效率","容量","功率"])) | (d["scenario"]=="基准")].copy(); d=d.drop_duplicates("scenario"); y=np.arange(len(d)); fig,axs=plt.subplots(1,2,figsize=(7.15,3.8)); delta=d["相对基准费用变化_pct"].to_numpy(); axs[0].barh(y,delta,color=[fs.COLORS["green"] if x<0 else fs.COLORS["red"] for x in delta],alpha=.82); axs[0].axvline(0,color=fs.COLORS["dark"],lw=.8); axs[0].set_yticks(y,d["scenario"],fontproperties=fs.CN); axs[0].set_xlabel("相对基准费用变化/%",fontproperties=fs.CN); fs.clean(axs[0])
    e=pd.read_csv(fs.RESULTS/"模型评价_误差倍率压力测试.csv"); base=pd.DataFrame([{"gamma":1,"相对基准费用变化_pct":0,"emergency_kwh":245483.293063}]); e=pd.concat([e,base]).sort_values("gamma"); axs[1].plot(e["gamma"],e["相对基准费用变化_pct"],marker="o",color=fs.COLORS["blue"],label="费用变化/%"); axs[1].plot(e["gamma"],e["emergency_kwh"]/10000,marker="s",color=fs.COLORS["orange"],label="应急购电/万kWh"); axs[1].set_xlabel(r"预测误差倍率 $\gamma$",fontproperties=fs.CN); axs[1].set_ylabel("相对量",fontproperties=fs.CN); axs[1].legend(); fs.clean(axs[1]); fs.panel_name(axs[0],"（a）物理参数敏感性"); fs.panel_name(axs[1],"（b）预测误差压力测试"); fig.subplots_adjust(bottom=.23,wspace=.30,left=.23); fs.save(fig,"sensitivity-overview")

def main(): fs.setup(); price_diagnostics(); outcome(); risk_stability(); sensitivity_overview()
if __name__=="__main__": main()
