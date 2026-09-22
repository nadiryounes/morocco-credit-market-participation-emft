import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
import pandas as pd
import os

# Ensure clean styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_fig1():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    
    # Define boxes
    boxes = {
        '2019': {
            'text': '2019 baseline\n\n• repeated cross-sectional analysis\n• ML training/development:\n  N=748 classifiable',
            'pos': (0.5, 0.85),
            'size': (0.6, 0.15)
        },
        'COVID': {
            'text': 'COVID follow-up rounds 2020–2021\n\n• R1, R2, R3\n• intermediate financial-stress measures',
            'pos': (0.5, 0.55),
            'size': (0.6, 0.15)
        },
        '2023': {
            'text': '2023 wave\n\n• repeated cross-sectional analysis\n• held-out ML evaluation:\n  N=529 classifiable',
            'pos': (0.5, 0.25),
            'size': (0.6, 0.15)
        }
    }
    
    # Draw central timeline boxes
    for k, v in boxes.items():
        box = patches.FancyBboxPatch(
            (v['pos'][0] - v['size'][0]/2, v['pos'][1] - v['size'][1]/2),
            v['size'][0], v['size'][1],
            boxstyle="round,pad=0.05",
            ec="#2b4750", fc="#eef4f7", lw=1.5
        )
        ax.add_patch(box)
        ax.text(v['pos'][0], v['pos'][1], v['text'], 
                ha='center', va='center', fontsize=11, fontweight='bold', color="#1a2d33")
    
    # Draw arrows
    ax.annotate('', xy=(0.5, 0.70), xytext=(0.5, 0.775),
                arrowprops=dict(facecolor='#2b4750', shrink=0, width=2, headwidth=8))
    ax.annotate('', xy=(0.5, 0.40), xytext=(0.5, 0.475),
                arrowprops=dict(facecolor='#2b4750', shrink=0, width=2, headwidth=8))
    
    # Side panel for linkages
    side_text = (
        "Matched Longitudinal Panel\n"
        "──────────────────────\n"
        "• 2019–2023 matched firms: N=280\n"
        "• Endpoint-classifiable firms: N=176\n"
        "• Strict complete-wave base: N=88\n\n"
        "COVID Construct-Valid Samples\n"
        "──────────────────────\n"
        "• Arrears sample: N=130\n"
        "• Liquidity sample: N=136"
    )
    
    box_side = patches.FancyBboxPatch(
        (0.1, 0.02), 0.8, 0.13,
        boxstyle="round,pad=0.02",
        ec="#5c796d", fc="#f2f7f5", lw=1.5
    )
    ax.add_patch(box_side)
    ax.text(0.5, 0.085, side_text, ha='center', va='center', fontsize=10, color="#2d4037")

    plt.savefig('Figure_1_Data_Architecture.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Figure_1_Data_Architecture.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig2():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['No financing need', 'Constrained non-applicant', 'Credit applicant']
    y2019 = [44.7, 36.1, 19.2]
    se2019 = [4.0, 3.8, 3.0]
    
    y2023 = [48.3, 35.9, 15.8]
    se2023 = [3.8, 3.7, 2.9]
    
    ci2019 = [1.96 * se for se in se2019]
    ci2023 = [1.96 * se for se in se2023]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, y2019, width, yerr=ci2019, label='2019 Baseline', 
           color='#4B8BBE', edgecolor='black', capsize=5, error_kw=dict(lw=1.5, capthick=1.5))
    ax.bar(x + width/2, y2023, width, yerr=ci2023, label='2023 Wave', 
           color='#FFE873', edgecolor='black', capsize=5, error_kw=dict(lw=1.5, capthick=1.5))
    
    ax.set_ylabel('Weighted Prevalence (%)', fontsize=11, fontweight='bold')
    ax.set_title('Survey-Weighted Credit-Market State Distributions (2019 vs 2023)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10)
    
    ax.set_ylim(0, 60)
    
    plt.savefig('Figure_2_Weighted_Credit_States.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Figure_2_Weighted_Credit_States.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig3():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    
    # Nodes: NFN, CNA, CA
    # 2019 totals: CA=26, CNA=56, NFN=94
    # 2023 totals: CA=32, CNA=50, NFN=94
    
    labels = ['Credit applicant', 'Constrained\nnon-applicant', 'No financing need']
    colors = ['#d7191c', '#fdae61', '#abdda4']
    
    h_2019 = [26, 56, 94]
    h_2023 = [32, 50, 94]
    
    y_2019 = [0]
    for h in h_2019[:-1]:
        y_2019.append(y_2019[-1] + h + 15) # gap of 15
        
    y_2023 = [0]
    for h in h_2023[:-1]:
        y_2023.append(y_2023[-1] + h + 15)
        
    # Flow matrix (2019 source -> 2023 target)
    # [CA, CNA, NFN]
    flows = [
        [7, 5, 14],   # From CA 2019
        [13, 11, 32], # From CNA 2019
        [12, 34, 48]  # From NFN 2019
    ]
    
    # Draw left nodes
    for i, (h, y) in enumerate(zip(h_2019, y_2019)):
        rect = patches.Rectangle((0, y), 0.5, h, facecolor=colors[i], edgecolor='black', alpha=0.9)
        ax.add_patch(rect)
        ax.text(-0.1, y + h/2, f"{labels[i]}\n(n={h})", ha='right', va='center', fontsize=11, fontweight='bold')
        
    # Draw right nodes
    for i, (h, y) in enumerate(zip(h_2023, y_2023)):
        rect = patches.Rectangle((9.5, y), 0.5, h, facecolor=colors[i], edgecolor='black', alpha=0.9)
        ax.add_patch(rect)
        ax.text(10.1, y + h/2, f"{labels[i]}\n(n={h})", ha='left', va='center', fontsize=11, fontweight='bold')

    ax.text(0.25, -20, '2019 Baseline', ha='center', fontsize=12, fontweight='bold')
    ax.text(9.75, -20, '2023 Wave', ha='center', fontsize=12, fontweight='bold')

    def draw_band(ax, x0, y0_bottom, h0, x1, y1_bottom, h1, color):
        path_data = [
            (mpath.Path.MOVETO, (x0, y0_bottom)),
            (mpath.Path.CURVE4, (x0 + 4, y0_bottom)),
            (mpath.Path.CURVE4, (x1 - 4, y1_bottom)),
            (mpath.Path.CURVE4, (x1, y1_bottom)),
            (mpath.Path.LINETO, (x1, y1_bottom + h1)),
            (mpath.Path.CURVE4, (x1 - 4, y1_bottom + h1)),
            (mpath.Path.CURVE4, (x0 + 4, y0_bottom + h0)),
            (mpath.Path.CURVE4, (x0, y0_bottom + h0)),
            (mpath.Path.CLOSEPOLY, (x0, y0_bottom))
        ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, edgecolor='none', alpha=0.4)
        ax.add_patch(patch)

    y_src_offsets = [0, 0, 0]
    y_tgt_offsets = [0, 0, 0]
    
    for i in range(3): # source
        for j in range(3): # target
            flow_val = flows[i][j]
            if flow_val > 0:
                y0_bot = y_2019[i] + y_src_offsets[i]
                y1_bot = y_2023[j] + y_tgt_offsets[j]
                draw_band(ax, 0.5, y0_bot, flow_val, 9.5, y1_bot, flow_val, colors[i])
                y_src_offsets[i] += flow_val
                y_tgt_offsets[j] += flow_val

    ax.set_xlim(-3, 13)
    ax.set_ylim(-30, max(y_2019[-1]+h_2019[-1], y_2023[-1]+h_2023[-1]) + 20)
    ax.invert_yaxis()
    
    plt.savefig('Figure_3_Credit_State_Transitions.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Figure_3_Credit_State_Transitions.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig4():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Internal versus External Temporal Predictive Performance', fontsize=14, fontweight='bold', y=0.98)
    
    models = ['LR', 'EBM', 'CatBoost']
    x = np.arange(len(models))
    width = 0.35
    
    # Data definitions
    metrics = {
        'Macro-F1 (Higher is better)': {
            'internal': [0.576, 0.577, 0.579],
            'external': [0.383, 0.356, 0.362],
            'ci_low': [0.341, 0.317, 0.318],
            'ci_high': [0.427, 0.399, 0.406],
            'ax': axes[0,0], 'ylim': (0.2, 0.65)
        },
        'Balanced Accuracy (Higher is better)': {
            'internal': [0.564, 0.565, 0.565],
            'external': [0.385, 0.365, 0.384],
            'ci_low': [0.348, 0.332, 0.352],
            'ci_high': [0.429, 0.403, 0.428],
            'ax': axes[0,1], 'ylim': (0.2, 0.65)
        },
        'LogLoss (Lower is better)': {
            'internal': [0.884, 0.863, 0.867],
            'external': [1.146, 1.094, 1.063],
            'ci_low': [1.066, 1.008, 0.985],
            'ci_high': [1.231, 1.171, 1.125],
            'ax': axes[1,0], 'ylim': (0.6, 1.4)
        },
        'Brier Score (Lower is better)': {
            'internal': [0.531, 0.515, 0.519],
            'external': [0.652, 0.634, 0.629],
            'ci_low': [0.611, 0.598, 0.589],
            'ci_high': [0.693, 0.671, 0.665],
            'ax': axes[1,1], 'ylim': (0.4, 0.8)
        }
    }
    
    for title, m in metrics.items():
        ax = m['ax']
        # Compute errors for external
        yerr_lower = [ext - cl for ext, cl in zip(m['external'], m['ci_low'])]
        yerr_upper = [ch - ext for ext, ch in zip(m['external'], m['ci_high'])]
        yerr = [yerr_lower, yerr_upper]
        
        ax.bar(x - width/2, m['internal'], width, label='Internal (2019 CV)', color='#4B8BBE', edgecolor='black')
        ax.bar(x + width/2, m['external'], width, yerr=yerr, label='External (2023 Held-out)', 
               color='#FFE873', edgecolor='black', capsize=5, error_kw=dict(lw=1.5, capthick=1.5))
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(m['ylim'])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if title == 'Macro-F1 (Higher is better)':
            ax.legend(fontsize=10, loc='upper right')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Figure_4_Temporal_Performance.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Figure_4_Temporal_Performance.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig5():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['LR', 'EBM', 'CatBoost']
    recall = [12.5, 6.2, 11.2]
    precision = [24.4, 21.7, 32.1]
    f1 = [16.5, 9.7, 16.7]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, recall, width, label='Recall', color='#1b9e77', edgecolor='black')
    ax.bar(x, precision, width, label='Precision', color='#d95f02', edgecolor='black')
    ax.bar(x + width, f1, width, label='F1 Score', color='#7570b3', edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Class-Specific External Transportability (CREDIT_APPLICANT)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    
    ax.set_ylim(0, 40)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    for container in ax.containers:
        add_labels(container)

    plt.savefig('Figure_5_Applicant_Transportability.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Figure_5_Applicant_Transportability.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
