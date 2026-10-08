"""Render figures from saved artifacts; requires matplotlib==3.10.7."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
pilot = json.loads((ROOT/'results/pilot/metrics.json').read_text())
review = json.loads((ROOT/'results/pilot/factual_review.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
losses = [pilot['baseline_test']['baseline_test_loss'],pilot['finetuned_test']['finetuned_test_loss']]
counts = [review['baseline_supported_answer_count'],review['finetuned_supported_answer_count']]
for ax, values, title, ymax, labels in [
    (axes[0],losses,'Lower held-out loss',3.8,[f'{v:.3f}' for v in losses]),
    (axes[1],counts,'Fewer supported answer bodies',10,[f'{v}/10' for v in counts])]:
    bars=ax.bar(['Base','QLoRA'],values,color=['#526785','#2d55c7'],width=.55)
    ax.bar_label(bars,labels=labels,padding=5)
    ax.set_ylim(0,ymax)
    ax.set_title(title,pad=14,fontweight='bold')
    ax.grid(axis='y',alpha=.16)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Assistant-token cross-entropy')
axes[1].set_ylabel('Strict agent-assisted support count')
fig.suptitle('Original ten-fact pilot: loss is not factual reliability',fontweight='bold')
fig.savefig(ROOT/'docs/pilot_results.png',dpi=180)
plt.close(fig)
summary_path=ROOT/'results/comparison/manual_metrics.json'
if summary_path.exists():
    data=json.loads(summary_path.read_text())
    methods=['base','qlora','rag_base','rag_qlora','gated_rag_base','extractive_bm25']
    labels=['Base','QLoRA','Base + RAG','QLoRA + RAG','Gated RAG','Extractive']
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    fig.get_layout_engine().set(rect=(0, .07, 1, .93))
    x=list(range(len(methods)))
    a=ax.bar([i-.18 for i in x],[data[m]['supported_answer_count'] for m in methods],width=.36,color='#2d55c7',label='Supported answer bodies / 10')
    b=ax.bar([i+.18 for i in x],[data[m]['appropriate_refusal_count'] for m in methods],width=.36,color='#21927e',label='Appropriate unsupported-question refusals / 10')
    ax.bar_label(a,padding=3);ax.bar_label(b,padding=3)
    ax.set_xticks(x,labels);ax.set_ylim(0,12.5);ax.set_yticks(range(0,11,2))
    ax.set_ylabel('Count (ten questions per category)')
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax.legend(loc='upper center',ncol=2,frameon=False)
    ax.set_title('Exploratory comparison: evidence access and abstention',fontweight='bold',pad=15)
    fig.text(.01,.005,'Agent-assisted labels; no independent human adjudication. RAG sees all 58 facts; closed-book fine-tuning sees 43.',fontsize=8)
    fig.savefig(ROOT/'docs/comparison_results.png',dpi=180)
    plt.close(fig)
