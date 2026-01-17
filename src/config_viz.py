"""
Pastel görselleştirme konfigürasyonu ve yardımcı fonksiyonlar.
Tüm matplotlib grafikleri için tutarlı pastel tema.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np


# ============================================================================
# PASTEL RENK PALETİ
# ============================================================================

PASTEL_COLORS = {
    'blue': '#A7C7E7',      # Açık Mavi
    'lavender': '#CDB4DB',  # Lavanta
    'baby_blue': '#BDE0FE', # Bebek Mavisi
    'pink': '#FFC8DD',      # Pembe
    'green': '#C8E6C9',     # Açık Yeşil
    'peach': '#FFDAB9',     # Şeftali
    'mint': '#B5EAD7',      # Mint
    'lilac': '#E0BBE4',     # Leylak
}

# Liste formatında (sıralı kullanım için)
PASTEL_PALETTE = [
    '#A7C7E7',  # Açık Mavi
    '#CDB4DB',  # Lavanta
    '#BDE0FE',  # Bebek Mavisi
    '#FFC8DD',  # Pembe
    '#C8E6C9',  # Açık Yeşil
    '#FFDAB9',  # Şeftali
    '#B5EAD7',  # Mint
    '#E0BBE4',  # Leylak
]

# Koyu tonlar (border, text için)
DARK_GRAY = '#333333'
MEDIUM_GRAY = '#666666'
LIGHT_GRAY = '#CCCCCC'
VERY_LIGHT_GRAY = '#F0F0F0'


# ============================================================================
# MATPLOTLIB GLOBAL AYARLARI
# ============================================================================

def setup_pastel_style():
    """
    Matplotlib için pastel temalı global stil ayarlarını uygular.
    Proje başlangıcında bir kez çağrılmalıdır.
    """
    # Figure ayarları
    rcParams['figure.facecolor'] = 'white'
    rcParams['figure.edgecolor'] = 'white'
    rcParams['figure.figsize'] = (10, 6)
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.facecolor'] = 'white'
    
    # Axes ayarları
    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.edgecolor'] = LIGHT_GRAY
    rcParams['axes.labelcolor'] = DARK_GRAY
    rcParams['axes.titlecolor'] = DARK_GRAY
    rcParams['axes.titlesize'] = 14
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.labelsize'] = 11
    rcParams['axes.linewidth'] = 1.0
    rcParams['axes.grid'] = True
    
    # Grid ayarları
    rcParams['grid.color'] = LIGHT_GRAY
    rcParams['grid.linestyle'] = '-'
    rcParams['grid.linewidth'] = 0.5
    rcParams['grid.alpha'] = 0.5
    
    # Text ayarları
    rcParams['text.color'] = DARK_GRAY
    rcParams['font.size'] = 10
    rcParams['font.family'] = 'sans-serif'
    
    # Tick ayarları
    rcParams['xtick.color'] = DARK_GRAY
    rcParams['ytick.color'] = DARK_GRAY
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    
    # Legend ayarları
    rcParams['legend.facecolor'] = 'white'
    rcParams['legend.edgecolor'] = LIGHT_GRAY
    rcParams['legend.framealpha'] = 0.8
    rcParams['legend.fontsize'] = 9
    
    print("✨ Pastel görselleştirme stili aktif edildi!")


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def get_pastel_color(index):
    """
    Verilen index için pastel palet rengini döndürür.
    Palette'ten fazla index gelirse döngüsel olarak devam eder.
    
    Args:
        index (int): Renk indexi
        
    Returns:
        str: Hex renk kodu
    """
    return PASTEL_PALETTE[index % len(PASTEL_PALETTE)]


def create_figure(figsize=(10, 6), title=None, grid=True):
    """
    Pastel temalı figure oluşturur.
    
    Args:
        figsize (tuple): Figure boyutu (genişlik, yükseklik)
        title (str): Figure başlığı (opsiyonel)
        grid (bool): Grid gösterilsin mi
        
    Returns:
        tuple: (fig, ax) matplotlib nesneleri
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    if grid:
        ax.grid(True, color=LIGHT_GRAY, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color=DARK_GRAY, pad=15)
    
    # Spine'ları hafif gri yap
    for spine in ax.spines.values():
        spine.set_edgecolor(LIGHT_GRAY)
        spine.set_linewidth(1.0)
    
    return fig, ax


def save_figure(fig, filepath, dpi=300):
    """
    Figure'ı yüksek kalitede kaydeder.
    
    Args:
        fig: matplotlib figure nesnesi
        filepath (str): Kayıt yolu
        dpi (int): Çözünürlük
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"📊 Görsel kaydedildi: {filepath}")


def create_pastel_barplot(data, labels, title, xlabel, ylabel, 
                          filepath=None, horizontal=False, sort_descending=False):
    """
    Pastel renkli bar chart oluşturur.
    
    Args:
        data (list): Veri değerleri
        labels (list): Bar etiketleri
        title (str): Grafik başlığı
        xlabel (str): X ekseni etiketi
        ylabel (str): Y ekseni etiketi
        filepath (str): Kayıt yolu (opsiyonel)
        horizontal (bool): Yatay bar chart mı
        sort_descending (bool): Azalan sırada sırala mı
        
    Returns:
        tuple: (fig, ax)
    """
    # Sıralama
    if sort_descending:
        sorted_pairs = sorted(zip(data, labels), reverse=True)
        data, labels = zip(*sorted_pairs)
    
    # Figure oluştur
    fig, ax = create_figure(figsize=(10, 6), title=title)
    
    # Renkler
    colors = [get_pastel_color(i) for i in range(len(data))]
    
    # Bar chart
    if horizontal:
        bars = ax.barh(labels, data, color=colors, edgecolor=DARK_GRAY, linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=11, color=DARK_GRAY)
        ax.set_ylabel(ylabel, fontsize=11, color=DARK_GRAY)
        
        # Değerleri bar'ların üzerine yaz
        for i, (bar, value) in enumerate(zip(bars, data)):
            ax.text(value, bar.get_y() + bar.get_height()/2, 
                   f' {value:.3f}' if isinstance(value, float) else f' {value}',
                   va='center', ha='left', fontsize=9, color=DARK_GRAY)
    else:
        bars = ax.bar(labels, data, color=colors, edgecolor=DARK_GRAY, linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=11, color=DARK_GRAY)
        ax.set_ylabel(ylabel, fontsize=11, color=DARK_GRAY)
        
        # X etiketlerini döndür
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Değerleri bar'ların üzerine yaz
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}' if isinstance(height, float) else f'{height}',
                   ha='center', va='bottom', fontsize=9, color=DARK_GRAY)
    
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
    
    return fig, ax


def create_info_box(text_lines, filepath=None, title="Bilgi", box_color=None):
    """
    Bilgi kutusu figürü oluşturur (metrik özeti vs. için).
    
    Args:
        text_lines (list): Metin satırları
        filepath (str): Kayıt yolu (opsiyonel)
        title (str): Başlık
        box_color (str): Kutu rengi (varsayılan: pastel mavi)
        
    Returns:
        fig: matplotlib figure
    """
    if box_color is None:
        box_color = PASTEL_COLORS['baby_blue']
    
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.axis('off')
    
    # Başlık
    ax.text(0.5, 0.95, title, transform=ax.transAxes,
           fontsize=16, fontweight='bold', color=DARK_GRAY,
           ha='center', va='top')
    
    # Dikdörtgen kutu
    rect = mpatches.FancyBboxPatch(
        (0.1, 0.1), 0.8, 0.75,
        boxstyle="round,pad=0.02",
        transform=ax.transAxes,
        facecolor=box_color,
        edgecolor=DARK_GRAY,
        linewidth=1.5,
        alpha=0.3
    )
    ax.add_patch(rect)
    
    # Metin satırları
    y_start = 0.75
    y_step = 0.6 / max(len(text_lines), 1)
    
    for i, line in enumerate(text_lines):
        y_pos = y_start - i * y_step
        ax.text(0.5, y_pos, line, transform=ax.transAxes,
               fontsize=11, color=DARK_GRAY,
               ha='center', va='center',
               family='monospace')
    
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
    
    return fig


def create_comparison_bars(data_dict, title, ylabel, filepath=None):
    """
    İki veya daha fazla grubu karşılaştıran grouped bar chart.
    
    Args:
        data_dict (dict): {'Grup1': [val1, val2], 'Grup2': [val1, val2]}
        title (str): Başlık
        ylabel (str): Y ekseni etiketi
        filepath (str): Kayıt yolu
        
    Returns:
        tuple: (fig, ax)
    """
    fig, ax = create_figure(title=title)
    
    groups = list(data_dict.keys())
    n_groups = len(groups)
    n_bars = len(data_dict[groups[0]])
    
    x = np.arange(n_bars)
    width = 0.8 / n_groups
    
    for i, group in enumerate(groups):
        offset = (i - n_groups/2 + 0.5) * width
        color = get_pastel_color(i)
        ax.bar(x + offset, data_dict[group], width, 
               label=group, color=color, edgecolor=DARK_GRAY, linewidth=0.5)
    
    ax.set_ylabel(ylabel, fontsize=11, color=DARK_GRAY)
    ax.set_xticks(x)
    ax.legend(framealpha=0.8, edgecolor=LIGHT_GRAY)
    
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
    
    return fig, ax


# ============================================================================
# OTOMATİK STİL AKTİVASYONU
# ============================================================================

# Bu modül import edildiğinde otomatik olarak pastel stili aktive et
setup_pastel_style()
