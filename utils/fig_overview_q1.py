import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figure_style as fs

def load_raw():
    a1 = pd.read_excel(fs.DATA / "附件1.xlsx")
    load = pd.read_excel(fs.DATA / "附件2.xlsx", sheet_name="小区负载").iloc[:, 1:].to_numpy(float)
    pv = pd.read_excel(fs.DATA / "附件2.xlsx", sheet_name="光伏发电实际功率").iloc[:, 1:].to_numpy(float)
    return a1, load, pv

def route():
    fig, ax = plt.subplots(figsize=(7.15, 3.15)); ax.axis("off")
    nodes = [(0.04,"附件1--4\n原始数据"),(0.23,"因果预测\n误差校正"),(0.42,"风险裕量\n日前合同"),
             (0.61,"日内更新\n滚动调度"),(0.80,"现金结算\n稳健性检验")]
    for x, txt in nodes:
        box = FancyBboxPatch((x, .43), .15, .23, boxstyle="round,pad=0.018,rounding_size=.025",
                             fc="#EEF5FA", ec=fs.COLORS["blue"], lw=1.15)
        ax.add_patch(box); ax.text(x+.075,.545,txt,ha="center",va="center",fontproperties=fs.CN)
    for i in range(4):
        ax.add_patch(FancyArrowPatch((nodes[i][0]+.15,.545),(nodes[i+1][0],.545),arrowstyle="-|>",
                                     mutation_scale=11,color=fs.COLORS["dark"],lw=1))
    branches=[(.115,.22,"问题一：确定性LP",fs.COLORS["green"]),(.495,.80,"问题二：历史预测+风险",fs.COLORS["orange"]),
              (.685,.22,"问题三：预报更新",fs.COLORS["purple"]),(.875,.80,"问题四：波动电价",fs.COLORS["red"])]
    for x,y,t,c in branches:
        ax.plot([x,x],[.43,y+.04],color=c,lw=1); ax.text(x,y,t,ha="center",va="center",color=c,fontproperties=fs.CN)
    fs.save(fig,"route")

def data_overview():
    a1, load, pv = load_raw(); t=np.arange(144)/6
    fig=plt.figure(figsize=(7.15,4.7)); gs=fig.add_gridspec(2,1,height_ratios=[1.15,1],hspace=.34)
    ax=fig.add_subplot(gs[0]); ax2=ax.twinx()
    l1=ax.plot(t,a1.iloc[:,2],label="负荷",color="#4C72B0",lw=1.4)
    l2=ax.plot(t,a1.iloc[:,3],label="光伏预测",color="#55A868",lw=1.35,ls="--")
    l3=ax2.step(t,a1.iloc[:,1],where="post",label="电价",color="#C44E52",lw=1.25)
    ax.set_ylabel("功率/kW",fontproperties=fs.CN); ax2.set_ylabel("电价/(元/kWh)",fontproperties=fs.CN,color="#C44E52")
    ax2.tick_params(axis="y",colors="#C44E52"); fs.time_axis(ax); fs.clean(ax); ax2.spines["top"].set_visible(False)
    handles=l1+l2+list(l3); ax.legend(handles,[h.get_label() for h in handles],ncol=3,loc="upper center")
    ah=fig.add_subplot(gs[1]); im=ah.imshow(pv,aspect="auto",origin="lower",extent=[0,24,1,365],cmap="YlOrBr")
    ah.set_xlabel("时刻/h",fontproperties=fs.CN); ah.set_xticks([0,6,12,18,24]); ah.set_ylabel("年内日序",fontproperties=fs.CN)
    cb=fig.colorbar(im,ax=ah,fraction=.025,pad=.025); cb.set_label("光伏实际功率/kW",fontproperties=fs.CN)
    fs.save(fig,"data-overview")

def q1_flow():
    fig,ax=plt.subplots(figsize=(6.7,3.15)); ax.axis("off")
    pts={"光伏":(.12,.72),"外部电网":(.12,.25),"交流母线":(.48,.49),"小区负荷":(.84,.72),"储能":(.84,.25)}
    colors=[fs.COLORS["green"],fs.COLORS["blue"],"#F5F5F5",fs.COLORS["orange"],fs.COLORS["purple"]]
    for (name,(x,y)),c in zip(pts.items(),colors):
        ax.add_patch(FancyBboxPatch((x-.09,y-.08),.18,.16,boxstyle="round,pad=.02",fc=c,alpha=.20,ec=c,lw=1.2)); ax.text(x,y,name,ha="center",va="center",fontproperties=fs.CN)
    arrows=[("光伏","交流母线","光伏利用"),("外部电网","交流母线","计划/应急购电"),("交流母线","小区负荷","供电"),("交流母线","储能",r"充电 $\eta_c$"),("储能","交流母线",r"放电 $\eta_d$")]
    for a,b,label in arrows:
        x1,y1=pts[a]; x2,y2=pts[b]; rad=.14 if (a,b)==("储能","交流母线") else 0
        ax.add_patch(FancyArrowPatch((x1+.10*np.sign(x2-x1),y1),(x2-.10*np.sign(x2-x1),y2),arrowstyle="-|>",mutation_scale=11,lw=1,color=fs.COLORS["dark"],connectionstyle=f"arc3,rad={rad}")); ax.text((x1+x2)/2,(y1+y2)/2+.05,label,ha="center",fontproperties=fs.CN,fontsize=8)
    ax.text(.84,.08,r"$S_{\min}\leq S_t\leq S_{\max},\quad 0\leq x_t,y_t\leq B_{\max}$",ha="center",fontsize=9)
    fs.save(fig,"q1-flow")

def q1_operation():
    d=pd.read_csv(fs.RESULTS/"问题1_逐槽计划.csv"); t=np.arange(len(d))/6
    fig,axs=plt.subplots(2,1,figsize=(7.15,4.55),sharex=True,gridspec_kw={"height_ratios":[1.45,1]})
    ax=axs[0]
    ax.plot(t,d["负荷_kWh"]*6,color="#333333",lw=1.35,label="负荷")
    ax.plot(t,d["光伏可用_kWh"]*6,color="#18A77B",lw=1.25,ls="--",label="光伏发电")
    ax.step(t,d["计划购电量_kWh"]*6,where="post",color="#1685D1",lw=1.25,label="计划购电")
    ax.set_ylabel("功率/kW",fontproperties=fs.CN); ax.legend(ncol=3,loc="upper center"); fs.clean(ax)
    ax=axs[1]; soc=d["槽末储电量_kWh"].to_numpy(); soc_full=np.r_[6000.0,soc]; t_soc=np.arange(145)/6
    ax.axhspan(10800,11250,color="#D9D9D9",alpha=.55); ax.axhspan(750,1200,color="#D9D9D9",alpha=.55)
    ax.plot(t_soc,soc_full,color="#18A77B",lw=1.45); ax.axhline(1200,ls="--",color=fs.COLORS["gray"],lw=.8); ax.axhline(10800,ls="--",color=fs.COLORS["gray"],lw=.8)
    ax.scatter([0,24],[soc_full[0],soc_full[-1]],s=18,color="#333333",zorder=3)
    ax.annotate(f"{soc_full[0]:.0f}",(0,soc_full[0]),xytext=(7,5),textcoords="offset points",fontsize=8)
    ax.annotate(f"{soc_full[-1]:.0f}",(24,soc_full[-1]),xytext=(-7,5),textcoords="offset points",ha="right",fontsize=8)
    ax.text(23.8,10800,"上限 10,800 kWh",ha="right",va="bottom",fontproperties=fs.CN,fontsize=7,color=fs.COLORS["gray"])
    ax.text(23.8,1200,"下限 1,200 kWh",ha="right",va="top",fontproperties=fs.CN,fontsize=7,color=fs.COLORS["gray"])
    ax.set_ylim(650,11350); ax.set_ylabel("储电量/kWh",fontproperties=fs.CN); fs.time_axis(ax); fs.clean(ax)
    fs.save(fig,"q1-operation")

def main():
    fs.setup(); route(); data_overview(); q1_flow(); q1_operation()
if __name__=="__main__": main()

