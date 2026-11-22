import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')

def setup_environment():
    os.makedirs('data', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams['axes.unicode_minus'] = False

def download_data():
    print("⏳ 데이터 다운로드 중 (크래프톤 vs 넷마블)...")
    krf = yf.download('259960.KS', start='2023-01-01', end='2025-01-01', auto_adjust=False)
    ntm = yf.download('251270.KS', start='2023-01-01', end='2025-01-01', auto_adjust=False)
    
    if isinstance(krf.columns, pd.MultiIndex):
        krf.columns = krf.columns.get_level_values(0)
    if isinstance(ntm.columns, pd.MultiIndex):
        ntm.columns = ntm.columns.get_level_values(0)
        
    krf.to_csv('data/krafton_data.csv')
    ntm.to_csv('data/netmarble_data.csv')
    return krf, ntm

def process_data(krf, ntm):
    for df in [krf, ntm]:
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['Disparity'] = ((df['Close'] - df['SMA20']) / df['SMA20']) * 100
        df['Vol20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(250)
    
    combined = pd.DataFrame({
        'Krafton_Close': krf['Close'],
        'Krafton_Return': krf['Daily_Return'],
        'Netmarble_Close': ntm['Close'],
        'Netmarble_Return': ntm['Daily_Return']
    }).ffill().bfill()
    
    combined['Krafton_Norm'] = (combined['Krafton_Close'] / combined['Krafton_Close'].iloc[0]) * 100
    combined['Netmarble_Norm'] = (combined['Netmarble_Close'] / combined['Netmarble_Close'].iloc[0]) * 100
    combined.to_csv('data/game_stocks_combined.csv')
    return krf, ntm, combined

def generate_visualizations(krf, ntm, combined):
    # 차트 1: 누적 수익률 & 실제 종가 비교
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    ax1.plot(combined.index, combined['Krafton_Norm'], label='크래프톤 (259960.KS)', color='#0047AB', linewidth=2)
    ax1.plot(combined.index, combined['Netmarble_Norm'], label='넷마블 (251270.KS)', color='#E056FD', linewidth=2)
    ax1.axhline(100, color='gray', linestyle='--', alpha=0.7, label='기준선 (100)')
    ax1.set_title('[차트 1-1] 크래프톤 vs 넷마블 누적 수익률 비교 (기준일 2023-01-02 = 100)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('정규화 지수 (기준=100)', fontsize=12)
    ax1.legend(loc='upper left')
    
    ax2_twin = ax2.twinx()
    p1 = ax2.plot(combined.index, combined['Krafton_Close'], color='#0047AB', label='크래프톤 종가 (KRW)', linewidth=1.8)
    p2 = ax2_twin.plot(combined.index, combined['Netmarble_Close'], color='#FF7675', label='넷마블 종가 (KRW)', linewidth=1.8)
    ax2.set_title('[차트 1-2] 종목별 실제 종가 추이 (이중 Y축)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('크래프톤 주가 (원)', color='#0047AB')
    ax2_twin.set_ylabel('넷마블 주가 (원)', color='#FF7675')
    ax2.yaxis.set_major_formatter('{x:,.0f}원')
    ax2_twin.yaxis.set_major_formatter('{x:,.0f}원')
    lines = p1 + p2
    ax2.legend(lines, [l.get_label() for l in lines], loc='upper left')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('images/01_price_trend_comparison.png', dpi=300)
    plt.close()

    # 차트 2: 크래프톤 20일 이동평균선 & 이격도
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(krf.index, krf['Close'], label='크래프톤 종가', color='#1E3799', linewidth=2)
    ax1.plot(krf.index, krf['SMA20'], label='20일 이동평균선 (SMA 20)', color='#FA983A', linewidth=2, linestyle='--')
    ax1.fill_between(krf.index, krf['Close'], krf['SMA20'], where=(krf['Close'] >= krf['SMA20']), color='#4CD137', alpha=0.2)
    ax1.fill_between(krf.index, krf['Close'], krf['SMA20'], where=(krf['Close'] < krf['SMA20']), color='#E84118', alpha=0.2)
    ax1.set_title('[차트 2] 크래프톤 종가 vs 20일 이동평균선(SMA 20)', fontsize=14, fontweight='bold')
    ax1.yaxis.set_major_formatter('{x:,.0f}원')
    ax1.legend(loc='upper left')

    ax2.plot(krf.index, krf['Disparity'], color='#2F3542', linewidth=1.5, label='20일 이격도 (%)')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.2)
    ax2.fill_between(krf.index, krf['Disparity'], 0, where=(krf['Disparity'] >= 0), color='#4CD137', alpha=0.3)
    ax2.fill_between(krf.index, krf['Disparity'], 0, where=(krf['Disparity'] < 0), color='#E84118', alpha=0.3)
    ax2.set_title('20일 이동평균선 이격도 (%)', fontsize=12, fontweight='bold')
    ax2.yaxis.set_major_formatter('{x:+.1f}%')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('images/02_krafton_sma20.png', dpi=300)
    plt.close()

    # 차트 3: 일일 수익률 & 롤링 변동성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax1.plot(krf.index, krf['Daily_Return'], label='크래프톤 일일 수익률(%)', color='#0984E3', alpha=0.7)
    ax1.plot(ntm.index, ntm['Daily_Return'], label='넷마블 일일 수익률(%)', color='#E84393', alpha=0.7)
    ax1.axhline(0, color='gray', linewidth=0.8)
    ax1.set_title('[차트 3-1] 크래프톤 vs 넷마블 일일 수익률 변동성 추이', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')

    ax2.plot(krf.index, krf['Vol20'], label=f'크래프톤 20일 연율화 변동성 (평균: {krf["Vol20"].mean():.1f}%)', color='#0984E3', linewidth=2)
    ax2.plot(ntm.index, ntm['Vol20'], label=f'넷마블 20일 연율화 변동성 (평균: {ntm["Vol20"].mean():.1f}%)', color='#E84393', linewidth=2)
    ax2.set_title('[차트 3-2] 20일 롤링 연율화 변동성 비교', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('images/03_daily_return_volatility.png', dpi=300)
    plt.close()

    # 차트 4: 분포 히스토그램 & 박스플롯
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(krf['Daily_Return'].dropna(), kde=True, color='#0984E3', label='크래프톤', stat="density", bins=40, alpha=0.4, ax=ax1)
    sns.histplot(ntm['Daily_Return'].dropna(), kde=True, color='#E84393', label='넷마블', stat="density", bins=40, alpha=0.4, ax=ax1)
    ax1.set_title('[차트 4-1] 일일 수익률 분포 히스토그램', fontsize=13, fontweight='bold')
    ax1.legend()

    df_returns = pd.DataFrame({
        '종목': ['크래프톤'] * len(krf['Daily_Return'].dropna()) + ['넷마블'] * len(ntm['Daily_Return'].dropna()),
        '일일수익률': list(krf['Daily_Return'].dropna()) + list(ntm['Daily_Return'].dropna())
    })
    sns.boxplot(x='종목', y='일일수익률', hue='종목', data=df_returns, palette=['#74b9ff', '#ff7675'], legend=False, ax=ax2, width=0.4)
    ax2.set_title('[차트 4-2] 일일 수익률 박스플롯', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/04_return_distribution.png', dpi=300)
    plt.close()
    print("✅ 차트 4종 images/ 폴더 저장 완료!")

if __name__ == '__main__':
    setup_environment()
    krf, ntm = download_data()
    krf, ntm, combined = process_data(krf, ntm)
    generate_visualizations(krf, ntm, combined)
    print(f"크래프톤 연율화 변동성: {krf['Daily_Return'].std()*np.sqrt(250):.2f}% | 최종 이격도: {krf['Disparity'].iloc[-1]:+.2f}%")
    print(f"넷마블 연율화 변동성: {ntm['Daily_Return'].std()*np.sqrt(250):.2f}% | 최종 이격도: {ntm['Disparity'].iloc[-1]:+.2f}%")