import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import optuna
import unicodedata
import warnings
import os
import base64
import plotly.express as px
import pickle
import re
import time

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI HORSE RACING SYSTEM", layout="wide")

# --- 💾 超強力：自動バックアップ＆復元機能 ---
def load_backup():
    try:
        if 'master_data' not in st.session_state and os.path.exists('backup_master.pkl'):
            st.session_state.master_data = pd.read_pickle('backup_master.pkl')
        if 'training_df' not in st.session_state and os.path.exists('backup_training.pkl'):
            st.session_state.training_df = pd.read_pickle('backup_training.pkl')
        if 'current_result' not in st.session_state and os.path.exists('backup_result.pkl'):
            st.session_state.current_result = pd.read_pickle('backup_result.pkl')
    except Exception: pass 

def save_backup():
    try:
        if 'master_data' in st.session_state: st.session_state.master_data.to_pickle('backup_master.pkl')
        if 'training_df' in st.session_state: st.session_state.training_df.to_pickle('backup_training.pkl')
        if 'current_result' in st.session_state: st.session_state.current_result.to_pickle('backup_result.pkl')
    except Exception: pass

load_backup()
if 'file_key' not in st.session_state: st.session_state.file_key = 0

# --- 🎨 デザインCSS & ヘッダー ---
img_name = "ferrari.png" 
BANNER_RED, BANNER_YELLOW, TEXT_COLOR = "#ff2800", "#ffca28", "#000000"

st.markdown("""
<style>
/* ⬇️ 赤く点滅する矢印とテキスト（ステップ①〜③用） */
@keyframes pulse-red {
    0%, 100% { transform: translateY(0); color: #ff2800; opacity: 1; }
    50% { transform: translateY(5px); color: #cc0000; opacity: 0.5; }
}
.guide-container {
    text-align: center;
    margin-bottom: 10px;
    background-color: #fff8f8;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #ffdcdc;
}
.guide-arrow {
    animation: pulse-red 1s infinite ease-in-out;
    font-size: 1.8rem;
    font-weight: bold;
    line-height: 1;
    margin-bottom: 5px;
}
.guide-text {
    font-weight: 900;
    color: #ff2800;
    font-size: 0.95rem;
}

/* ↗️ 右上を指す、黒/濃い赤の点滅サイン（ステップ④用・眩しすぎない） */
@keyframes pulse-dark {
    0%, 100% { color: #770000; opacity: 1; }
    50% { color: #000000; opacity: 0.6; }
}
.blinking-close {
    animation: pulse-dark 1.5s infinite ease-in-out;
    font-size: 1.05rem;
    font-weight: 900;
    text-align: right;
    margin-top: -5px;
    margin-bottom: 15px;
    padding: 10px 15px;
    background-color: #f8f9fa;
    border-radius: 5px;
    border-right: 5px solid #ff2800;
}

.streamlit-expanderHeader { font-weight: bold; color: #333 !important; background-color: #f8f9fa; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

def get_banner():
    img_html = f"<img src='data:image/png;base64,{base64.b64encode(open(img_name, 'rb').read()).decode()}' style='width: 100%; max-width: 380px;'>" if os.path.exists(img_name) else "<h1>🏇</h1>"
    return f"""<div style="background-color: {BANNER_RED}; padding: 15px; border-radius: 15px; margin-bottom: 20px;">
        <div style="background-color: {BANNER_YELLOW}; border-radius: 10px; padding: 20px 30px; display: flex; align-items: center;">
            <div style="flex: 1; text-align: center;">{img_html}</div>
            <div style="flex: 2; padding-left: 30px;">
                <h1 translate='no' class='notranslate' style='color: {TEXT_COLOR}; font-family: Arial; font-size: 3.2rem; font-weight: 900; margin: 0;'>AI HORSE RACING SYSTEM</h1>
                <p style='color: {TEXT_COLOR}; font-size: 1.1rem; font-weight: bold;'>ULTIMATE STACKING & OPTUNA ENGINE</p>
            </div>
        </div></div>"""

st.markdown(get_banner(), unsafe_allow_html=True)

# --- 🧠 階層型メタAIエンジン ---
@st.cache_resource
def load_and_train_ai():
    file_path = '5yers_data.zip' if os.path.exists('5yers_data.zip') else '5yers_data.csv'
    try: df_past = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
    except: df_past = pd.read_csv(file_path, encoding='cp932', on_bad_lines='skip', engine='python')

    df_past = df_past.rename(columns={'着順': '確定着順', '天気': '天候'})
    df_past['確定着順_num'] = df_past['確定着順'].apply(lambda x: int("".join([c for c in unicodedata.normalize('NFKC', str(x)) if c.isdigit()])) if any(c.isdigit() for c in str(x)) else 99)
    df_past['is_top3'] = (df_past['確定着順_num'] <= 3).astype(int)

    df_past = df_past.sort_values(['馬名', '日付'] if '日付' in df_past.columns else ['馬名'])
    df_past['前走距離'] = df_past.groupby('馬名')['距離'].shift(1)
    df_past['距離増減'] = df_past['距離'] - df_past['前走距離'].fillna(df_past['距離'])
    df_past['継続騎乗'] = (df_past['騎手'] == df_past.groupby('馬名')['騎手'].shift(1)).astype(int)

    style_map = {'逃げ': 1, '先行': 2, '中団': 3, '後方': 4, 'ﾏｸﾘ': 5}
    df_past['脚質_num'] = df_past['脚質'].astype(str).str.strip().map(style_map).fillna(3)
    df_past['馬番'] = pd.to_numeric(df_past['馬番'], errors='coerce')
    df_past['芝ダ_str'] = df_past['芝・ダ'].astype(str).str.strip().str.replace('田', 'ダ')
    
    df_past['斤量'] = pd.to_numeric(df_past['斤量'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    df_past['馬体重'] = pd.to_numeric(df_past['馬体重'], errors='coerce')
    df_past['斤量比'] = (df_past['斤量'] / df_past['馬体重']).fillna(0)
    
    race_col_past = 'R' if 'R' in df_past.columns else ('レース' if 'レース' in df_past.columns else None)
    if '日付' in df_past.columns and race_col_past:
        nige_count = df_past[df_past['脚質_num'] == 1].groupby(['日付', '場所', race_col_past]).size().reset_index(name='逃げ馬頭数')
        df_past = pd.merge(df_past, nige_count, on=['日付', '場所', race_col_past], how='left')
    else: df_past['逃げ馬頭数'] = 0
    df_past['逃げ馬頭数'] = df_past['逃げ馬頭数'].fillna(0)

    j_stats = df_past.groupby(['場所', '騎手'])['is_top3'].mean().reset_index().rename(columns={'is_top3': '場所別騎手複勝率'})
    s_stats = df_past.groupby(['場所', '種牡馬'])['is_top3'].mean().reset_index().rename(columns={'is_top3': '場所別血統複勝率'})
    g_stats = df_past.groupby(['場所', '芝ダ_str', '馬番'])['is_top3'].mean().reset_index().rename(columns={'is_top3': 'コース別馬番複勝率'})
    horse_agg = df_past.groupby('馬名').agg(過去平均上り3F=('上り3F', 'mean'), 最終走距離=('距離', 'last'), 最終走騎手=('騎手', 'last')).reset_index()
    bs_stats = df_past.groupby(['場所', '母父'])['is_top3'].mean().reset_index().rename(columns={'is_top3': '場所別母父複勝率'}) if '母父' in df_past.columns else pd.DataFrame()
    tj_stats = df_past.groupby(['調教師', '騎手'])['is_top3'].mean().reset_index().rename(columns={'is_top3': '調教師_騎手_複勝率'}) if '調教師' in df_past.columns else pd.DataFrame()

    df_past = pd.merge(df_past, j_stats, on=['場所', '騎手'], how='left')
    df_past = pd.merge(df_past, s_stats, on=['場所', '種牡馬'], how='left')
    df_past = pd.merge(df_past, g_stats, on=['場所', '芝ダ_str', '馬番'], how='left')
    df_past = pd.merge(df_past, horse_agg, on='馬名', how='left')
    if not bs_stats.empty: df_past = pd.merge(df_past, bs_stats, on=['場所', '母父'], how='left')
    if not tj_stats.empty: df_past = pd.merge(df_past, tj_stats, on=['調教師', '騎手'], how='left')

    weather_map = {'晴':1, '曇':2, '小雨':3, '雨':4, '小雪':5, '雪':6}
    track_map = {'良':1, '稍':2, '稍重':2, '重':3, '不':4, '不良':4}
    surface_map = {'芝':1, 'ダ':2, '田':2, '障':3}
    place_map = {p: i for i, p in enumerate(df_past['場所'].unique())}

    df_past['天候_num'] = df_past['天候'].astype(str).str.strip().map(weather_map)
    df_past['馬場状態_num'] = df_past['馬場状態'].astype(str).str.strip().map(track_map)
    df_past['芝ダ_num'] = df_past['芝ダ_str'].map(surface_map)
    df_past['場所_num'] = df_past['場所'].map(place_map)
    df_past['距離'] = pd.to_numeric(df_past['距離'], errors='coerce')
    df_past['年齢'] = pd.to_numeric(df_past['年齢'], errors='coerce')

    train_features = ['馬番', '斤量', '年齢', '馬体重', '場所別騎手複勝率', '場所別血統複勝率', '過去平均上り3F', 
                      '天候_num', '馬場状態_num', '芝ダ_num', '距離', '場所_num', 'コース別馬番複勝率', 
                      '脚質_num', '距離増減', '継続騎乗', '斤量比', '逃げ馬頭数']
    
    if '場所別母父複勝率' in df_past.columns: train_features.append('場所別母父複勝率')
    if '調教師_騎手_複勝率' in df_past.columns: train_features.append('調教師_騎手_複勝率')

    for col in train_features: df_past[col] = pd.to_numeric(df_past[col], errors='coerce').fillna(0)
    X = df_past[train_features]
    y = df_past['is_top3']

    lgb_params = {'n_estimators': 100, 'random_state': 42, 'learning_rate': 0.05, 'verbose': -1}
    if os.path.exists('best_params.pkl'):
        try:
            with open('best_params.pkl', 'rb') as f: lgb_params.update(pickle.load(f))
        except: pass

    with st.spinner("🧙 メタAIを含む階層型学習を実行中..."):
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42, learning_rate=0.05, eval_metric='logloss')
        m_cat = CatBoostClassifier(iterations=100, random_state=42, learning_rate=0.05, verbose=0)
        
        estimators = [('lgb', m_lgb), ('xgb', m_xgb), ('cat', m_cat)]
        meta_ai = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
        meta_ai.fit(X, y)

    return meta_ai, j_stats, s_stats, g_stats, bs_stats, tj_stats, horse_agg, place_map, weather_map, track_map, surface_map, train_features, style_map, X, y

# --- ヘルパー関数 ---
def read_uploaded_file(uploaded_file, is_syutuba=False):
    uploaded_file.seek(0)
    kwargs = {'engine': 'python'}
    if is_syutuba: kwargs['header'] = None
    else: kwargs['on_bad_lines'] = 'skip'
    try: df = pd.read_csv(uploaded_file, encoding='utf-8', **kwargs)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='cp932', **kwargs)
    return df

def calc_dev(series):
    series = pd.to_numeric(series, errors='coerce')
    s_clean = series.dropna()
    if len(s_clean) < 2: return pd.Series([50.0]*len(series), index=series.index)
    return -(series - s_clean.mean()) / (s_clean.std() + 0.001) * 10 + 50

def get_waku_color(val):
    try: val = int(val)
    except: return ''
    colors = {1:'#f8f9fa', 2:'#343a40', 3:'#e63946', 4:'#0077b6', 5:'#ffd166', 6:'#2a9d8f', 7:'#f77f00', 8:'#ffb5a7'}
    return colors.get(val, '')

def clean_zougen_str(x):
    if pd.isna(x) or str(x).strip() in ['', '未定', '計不']: return np.nan
    s = str(x).replace(' ', '').replace('　', '') 
    try: return float(s)
    except: return np.nan

(meta_ai, j_stats, s_stats, g_stats, bs_stats, tj_stats, horse_agg, place_map, weather_map, track_map, surface_map, train_features, style_map, X_train, y_train) = load_and_train_ai()

# --- サイドバー ---
with st.sidebar:
    is_master_ready = 'master_data' in st.session_state
    has_tuned = os.path.exists('best_params.pkl')
    syutuba_file_raw = st.session_state.get(f"syutuba_{st.session_state.file_key}")
    syutuba_uploaded = bool(syutuba_file_raw)
    
    # 🏁 誘導ガイド (4) 準備完了（右上の閉じるボタンを案内）
    if is_master_ready and has_tuned:
        st.markdown('<div class="blinking-close">④ 準備完了！<br>右上の「×」または「>」でメニューを閉じてください ↗️</div>', unsafe_allow_html=True)
    
    st.header("⚙️ データ読み込み")
    
    # 🏁 誘導ガイド (1) ファイルアップロード
    if not is_master_ready and not syutuba_uploaded:
        st.markdown('<div class="guide-container"><div class="guide-arrow">⬇️</div><div class="guide-text">①出馬表（csv）調教データ（csv）を入力後②へ<br>（※出馬表だけでも可）</div></div>', unsafe_allow_html=True)

    syutuba_file = st.file_uploader("📂 出馬表 (CSV)", type=["csv"], key=f"syutuba_{st.session_state.file_key}")
    training_files = st.file_uploader("📂 調教データ (CSV)", type=["csv"], accept_multiple_files=True, key=f"training_{st.session_state.file_key}")
    
    # 🏁 誘導ガイド (2) 記憶させる
    if not is_master_ready and syutuba_uploaded:
        st.markdown('<div class="guide-container"><div class="guide-arrow">⬇️</div><div class="guide-text">②</div></div>', unsafe_allow_html=True)
        
    save_btn_type = "primary" if (not is_master_ready and syutuba_uploaded) else "secondary"
    if st.button("💾 ファイルをシステムに記憶させる", type=save_btn_type, use_container_width=True):
        if syutuba_file:
            df_raw = read_uploaded_file(syutuba_file, is_syutuba=True)
            df_all = df_raw.iloc[:, :24].copy()
            df_all.columns = ['枠番', '馬番', '場所', 'R', 'レース名', '芝ダ', '距離', '頭数', '馬名', '時刻', '条件', 'B', '性別', '年齢', '騎手', '斤量', '馬体重', '増減', '所属', '調教師', '父', '母父', '前走着順', '脚質']
            df_all['レース名'] = df_all['レース名'].astype(str).str.replace('クラッス', 'クラス').str.replace('クラックス', 'クラス').str.replace('ｸﾗｽ', 'クラス')
            df_all['レース名'] = df_all['レース名'].apply(lambda x: unicodedata.normalize('NFKC', x))
            df_all['芝ダ'] = df_all['芝ダ'].astype(str).str.strip().str.replace('田', 'ダ')
            df_all['馬番'] = pd.to_numeric(df_all['馬番'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
            df_all['枠番'] = pd.to_numeric(df_all['枠番'], errors='coerce')
            df_all['馬名'] = df_all['馬名'].astype(str).str.strip()
            df_all['馬体重'] = pd.to_numeric(df_all['馬体重'], errors='coerce')
            df_all['増減'] = df_all['増減'].apply(clean_zougen_str)
            df_all['R'] = pd.to_numeric(df_all['R'], errors='coerce').fillna(0).astype(int)
            st.session_state.master_data = df_all
            save_backup()
            st.success("✅ 出馬表を記憶しました！")
            
        if training_files:
            dfs_t = [read_uploaded_file(f) for f in training_files]
            st.session_state.training_df = pd.concat(dfs_t, ignore_index=True)
            save_backup()
            st.success("✅ 調教データを記憶しました！")
        st.rerun()

    st.markdown("---")
    
    # 🏁 誘導ガイド (3) チューニング
    if is_master_ready and not has_tuned:
        st.markdown('<div class="guide-container"><div class="guide-arrow">⬇️</div><div class="guide-text">③</div></div>', unsafe_allow_html=True)
        
    if st.button("🎯 Optunaチューニング実行", type="primary" if (is_master_ready and not has_tuned) else "secondary", use_container_width=True):
        with st.spinner("Optunaが最適なパラメータを探索中..."):
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'max_depth': trial.suggest_int('max_depth', 3, 10)
                }
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                return model.score(X_train, y_train)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=5)
            
            with open('best_params.pkl', 'wb') as f:
                pickle.dump(study.best_params, f)
            st.cache_resource.clear()
        st.success("✅ 最適化完了！次回予測から適用されます。")
        time.sleep(2)
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ オールクリア (全データ初期化)", type="secondary", use_container_width=True):
        new_key = st.session_state.get('file_key', 0) + 1
        st.session_state.clear()
        st.session_state['file_key'] = new_key
        for f in ['backup_master.pkl', 'backup_training.pkl', 'backup_result.pkl', 'best_params.pkl']:
            if os.path.exists(f): 
                try: os.remove(f)
                except: pass
        st.rerun()

# --- AI予測ロジック ---
def run_analysis(input_df):
    df_work = input_df.copy()
    df_work['脚質_num'] = df_work['脚質'].astype(str).str.strip().map(style_map).fillna(3)
    df_work['前走確定着順'] = pd.to_numeric(df_work['前走着順'], errors='coerce').fillna(10)
    
    df_work['斤量'] = pd.to_numeric(df_work['斤量'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    df_work['馬体重'] = pd.to_numeric(df_work['馬体重'], errors='coerce')
    df_work['斤量比'] = (df_work['斤量'] / df_work['馬体重']).fillna(0)
    
    if 'R' in df_work.columns:
        nige_count = df_work[df_work['脚質_num'] == 1].groupby(['場所', 'R']).size().reset_index(name='逃げ馬頭数')
        df_work = pd.merge(df_work, nige_count, on=['場所', 'R'], how='left')
    else: df_work['逃げ馬頭数'] = 0
    df_work['逃げ馬頭数'] = df_work['逃げ馬頭数'].fillna(0)
    
    df_work = pd.merge(df_work, horse_agg, on='馬名', how='left')
    df_work['距離増減'] = pd.to_numeric(df_work['距離'], errors='coerce') - df_work['最終走距離'].fillna(pd.to_numeric(df_work['距離'], errors='coerce'))
    df_work['継続騎乗'] = (df_work['騎手'] == df_work['最終走騎手']).astype(int)

    df_work = pd.merge(df_work, j_stats, on=['場所', '騎手'], how='left')
    df_work = pd.merge(df_work, s_stats, left_on=['場所', '父'], right_on=['場所', '種牡馬'], how='left')
    df_work = pd.merge(df_work, g_stats, left_on=['場所', '芝ダ', '馬番'], right_on=['場所', '芝ダ_str', '馬番'], how='left')
    if not bs_stats.empty: df_work = pd.merge(df_work, bs_stats, on=['場所', '母父'], how='left')
    if not tj_stats.empty: df_work = pd.merge(df_work, tj_stats, on=['調教師', '騎手'], how='left')

    df_work['芝ダ_num'] = df_work['芝ダ'].map(surface_map)
    df_work['天候_num'] = np.nan
    df_work['馬場状態_num'] = np.nan
    for venue in df_work['場所'].unique():
        w_val = st.session_state.get(f"weather_{venue}", "指定なし")
        t_shiba_val = st.session_state.get(f"track_shiba_{venue}", "指定なし")
        t_dirt_val = st.session_state.get(f"track_dirt_{venue}", "指定なし")
        idx_venue = df_work['場所'] == venue
        idx_shiba = idx_venue & (df_work['芝ダ_num'] == 1)
        idx_dirt = idx_venue & (df_work['芝ダ_num'] != 1) 
        df_work.loc[idx_venue, '天候_num'] = weather_map.get(w_val, np.nan)
        df_work.loc[idx_shiba, '馬場状態_num'] = track_map.get(t_shiba_val, np.nan)
        df_work.loc[idx_dirt, '馬場状態_num'] = track_map.get(t_dirt_val, np.nan)

    df_work['場所_num'] = df_work['場所'].map(place_map)
    df_work['距離'] = pd.to_numeric(df_work['距離'], errors='coerce')
    df_work['年齢'] = pd.to_numeric(df_work['年齢'], errors='coerce')

    X = df_work[train_features].fillna(0)
    
    # --- 🧙 メタAIによる統合予測 ---
    df_work['AI予測_複勝確率'] = (meta_ai.predict_proba(X)[:, 1] * 100).round(1)

    df_work['調教スコア'] = 50.0
    if 'training_df' in st.session_state:
        df_training = st.session_state.training_df.copy()
        if '馬名' in df_training.columns and 'Lap1' in df_training.columns:
            df_training['馬名'] = df_training['馬名'].astype(str).str.strip()
            df_training['Lap1'] = pd.to_numeric(df_training['Lap1'], errors='coerce')
            df_training['Lap2'] = pd.to_numeric(df_training['Lap2'], errors='coerce')
            df_training['加速'] = df_training['Lap1'] - df_training['Lap2']
            course_col = '調教種別' if '調教種別' in df_training.columns else ('コース' if 'コース' in df_training.columns else None)
            if course_col:
                t_agg = df_training.groupby([course_col, '馬名']).agg(L1=('Lap1','mean'), K=('加速','mean')).reset_index()
                score_list = []
                for course, group in t_agg.groupby(course_col):
                    g = group.copy()
                    g['L1_dev'] = calc_dev(g['L1'])
                    g['K_dev'] = calc_dev(g['K'])
                    g['コース別スコア'] = (g['L1_dev'] + g['K_dev']) / 2
                    score_list.append(g)
                if score_list:
                    t_scored = pd.concat(score_list)
                    final_t = t_scored.groupby('馬名')['コース別スコア'].max().reset_index().rename(columns={'コース別スコア': 'temp_score'})
                    df_work = pd.merge(df_work, final_t, on='馬名', how='left')
                    df_work['調教スコア'] = df_work['temp_score'].fillna(50.0)
                    df_work = df_work.drop(columns=['temp_score'])
            else:
                train_agg = df_training.groupby('馬名').agg(L1=('Lap1','mean'), K=('加速','mean')).reset_index()
                df_work = pd.merge(df_work, train_agg, on='馬名', how='left')
                df_work['調教スコア'] = (calc_dev(df_work['L1']) + calc_dev(df_work['K'])) / 2

    df_work['調教スコア'] = df_work['調教スコア'].fillna(50.0).round(1)

    # --- 🎛️ 当日の馬場バイアス補正計算 ---
    venue_str = df_work['場所'].iloc[0] if not df_work.empty else ""
    bias_style = st.session_state.get(f"bias_style_{venue_str}", "フラット")
    bias_waku = st.session_state.get(f"bias_waku_{venue_str}", "フラット")
    
    style_bonus_map = {"逃げ": 1, "先行": 2, "フラット": 3, "中団": 4, "後方": 5}
    waku_bonus_map = {"内枠": 2, "やや内": 4, "フラット": 0, "やや外": 6, "外枠": 8}
    
    style_weight = 0 if bias_style == "フラット" else 2.0
    waku_weight = 0 if bias_waku == "フラット" else 1.0
    
    target_style = style_bonus_map.get(bias_style, 3)
    target_waku = waku_bonus_map.get(bias_waku, 4)
    
    df_work['バイアス補正'] = 0.0
    if style_weight > 0: df_work['バイアス補正'] += (4 - abs(df_work['脚質_num'] - target_style)) * style_weight
    if waku_weight > 0: df_work['バイアス補正'] += (8 - abs(df_work['馬番'] - target_waku)) * (waku_weight / 2)

    df_work['総合AIスコア'] = ((df_work['AI予測_複勝確率'] * 0.70) + (df_work['調教スコア'] * 0.20) + df_work['バイアス補正']).round(1)
    df_work['コース別馬番複勝率_disp'] = (df_work['コース別馬番複勝率'] * 100).round(1)
    
    final = []
    for _, g in df_work.groupby(['場所', 'R', 'レース名']):
        g = g.sort_values('総合AIスコア', ascending=False).reset_index(drop=True)
        g['予想印'] = [('◎' if i==0 else '〇' if i==1 else '▲' if i==2 else '△' if i<=5 else '') for i in range(len(g))]
        final.append(g)
    return pd.concat(final)


# --- メイン画面 ---
tab1, tab2, tab3 = st.tabs(["🔮 AI PREDICTION (予想)", "📊 PERFORMANCE (成績分析)", "📖 MANUAL (説明書)"])

with tab1:
    if st.session_state.get('predicted_just_now'):
        st.toast("✨ 予想を最新データで更新しました！", icon="✅")
        st.session_state.predicted_just_now = False

    if st.session_state.get('weight_updated_just_now'):
        st.toast(f"⚖️ {st.session_state.weight_updated_just_now} の馬体重を反映しました！", icon="✅")
        st.session_state.weight_updated_just_now = False

    if 'master_data' in st.session_state:
        st.markdown("### 🏁 レース選択 & 条件設定")
        c1, c2 = st.columns([1.5, 1.5])
        st.session_state.master_data['R'] = pd.to_numeric(st.session_state.master_data['R'], errors='coerce').fillna(0).astype(int)
        venues = st.session_state.master_data['場所'].unique()
        with c1: selected_venue = st.selectbox("📍 競馬場", venues, key="sel_venue")
        
        races = st.session_state.master_data[st.session_state.master_data['場所'] == selected_venue]['R'].unique()
        with c2: selected_race = st.selectbox("🏁 R", sorted(races), key="sel_race")

        with st.expander("⛅ 天候・馬場状態の設定 (タップで開閉)", expanded=False):
            wc1, wc2, wc3 = st.columns(3)
            w_options, t_options = ["指定なし", "晴", "曇", "小雨", "雨", "小雪", "雪"], ["指定なし", "良", "稍重", "重", "不良"]
            saved_w = st.session_state.get(f"weather_{selected_venue}", "指定なし")
            saved_ts = st.session_state.get(f"track_shiba_{selected_venue}", "指定なし")
            saved_td = st.session_state.get(f"track_dirt_{selected_venue}", "指定なし")
            
            with wc1: st.session_state[f"weather_{selected_venue}"] = st.selectbox("天候", w_options, index=w_options.index(saved_w) if saved_w in w_options else 0, key=f"w_widget_{selected_venue}")
            with wc2: st.session_state[f"track_shiba_{selected_venue}"] = st.selectbox("芝", t_options, index=t_options.index(saved_ts) if saved_ts in t_options else 0, key=f"ts_widget_{selected_venue}")
            with wc3: st.session_state[f"track_dirt_{selected_venue}"] = st.selectbox("ダート", t_options, index=t_options.index(saved_td) if saved_td in t_options else 0, key=f"td_widget_{selected_venue}")

        with st.expander("🎛️ 本日の馬場バイアス補正", expanded=False):
            st.info("当日のレース傾向（前残り、内枠有利など）を最終スコアに加味します。")
            bc1, bc2 = st.columns(2)
            with bc1:
                bias_style = st.select_slider("🔥 有利な脚質", options=["逃げ", "先行", "フラット", "中団", "後方"], value=st.session_state.get(f"bias_style_{selected_venue}", "フラット"), key=f"bs_{selected_venue}")
                st.session_state[f"bias_style_{selected_venue}"] = bias_style
            with bc2:
                bias_waku = st.select_slider("🛤️ 有利な枠", options=["内枠", "やや内", "フラット", "やや外", "外枠"], value=st.session_state.get(f"bias_waku_{selected_venue}", "フラット"), key=f"bw_{selected_venue}")
                st.session_state[f"bias_waku_{selected_venue}"] = bias_waku

        st.markdown("---")

        race_df = st.session_state.master_data[(st.session_state.master_data['場所'] == selected_venue) & (st.session_state.master_data['R'] == selected_race)].copy()
        race_df['馬番'] = pd.to_numeric(race_df['馬番'], errors='coerce')
        race_df = race_df.sort_values('馬番').reset_index(drop=True)
        
        raw_race_name = str(race_df['レース名'].iloc[0]) if not race_df.empty else ""
        clean_name = re.sub(r'第?\d+回|\d+日目?|[^\s]*競馬場|\d+[RＲrｒ]|^[\s\-ー_]+', '', raw_race_name).replace(selected_venue, '').strip()

        course_type = str(race_df['芝ダ'].iloc[0]).replace('田', 'ダ') if not race_df.empty else ""
        course_dist = str(int(pd.to_numeric(race_df['距離'].iloc[0], errors='coerce'))) if not race_df.empty and pd.notna(race_df['距離'].iloc[0]) else ""
        
        w_val_disp = st.session_state.get(f"weather_{selected_venue}", "指定なし")
        t_val_disp = st.session_state.get(f"track_shiba_{selected_venue}", "指定なし") if course_type == '芝' else st.session_state.get(f"track_dirt_{selected_venue}", "指定なし")

        st.subheader(f"🏆 {selected_venue} {int(selected_race)}R" + (f" - {clean_name}" if clean_name and clean_name.lower() != "nan" else ""))
        st.markdown(f"<h5 style='color: #444;'>🛣️ {course_type}{course_dist}m ｜ ⛅ 天候: <b>{w_val_disp}</b> ｜ 💧 馬場: <b>{t_val_disp}</b> ｜ 🎛️ バイアス: <b>{bias_style}/{bias_waku}</b></h5>", unsafe_allow_html=True)

        if 'current_result' in st.session_state:
            pred_df = st.session_state.current_result[(st.session_state.current_result['場所'] == selected_venue) & (st.session_state.current_result['R'] == selected_race)]
            if not pred_df.empty:
                merge_cols = ['馬名', '予想印', '総合AIスコア', 'AI予測_複勝確率', '調教スコア']
                if 'コース別馬番複勝率_disp' in pred_df.columns: merge_cols.append('コース別馬番複勝率_disp')
                race_df = pd.merge(race_df, pred_df[merge_cols], on='馬名', how='left')
                if 'コース別馬番複勝率_disp' in race_df.columns: race_df = race_df.rename(columns={'コース別馬番複勝率_disp': '馬番複勝率'})
            else: race_df['予想印'] = ""
        else: race_df['予想印'] = ""

        disp_cols = ['予想印', '枠番', '馬番', '馬名', '斤量', '騎手', '馬体重', '増減']
        if 'AI予測_複勝確率' in race_df.columns: disp_cols.extend(['AI予測_複勝確率', '調教スコア', '総合AIスコア'])
        if '馬番複勝率' in race_df.columns: disp_cols.append('馬番複勝率')

        disp = race_df[disp_cols].copy()

        def apply_waku(row):
            w = int(row['枠番']) if pd.notna(row['枠番']) else 0
            c = get_waku_color(w)
            txt = 'white' if w in [2,3,4,6,7] else 'black'
            return [f'background-color: {c}; color: {txt}' if col=='馬番' else '' for col in row.index]

        styled_disp = disp.style.apply(apply_waku, axis=1)

        edited = st.data_editor(
            styled_disp,
            column_config={
                "予想印": st.column_config.Column("予想印", disabled=True), "枠番": None, 
                "馬番": st.column_config.NumberColumn("馬番", disabled=True), "馬名": st.column_config.Column("馬名", disabled=True),
                "斤量": st.column_config.NumberColumn("斤量", disabled=True), "騎手": st.column_config.Column("騎手", disabled=True),
                "馬体重": st.column_config.NumberColumn("馬体重", min_value=300, max_value=700, step=2), "増減": st.column_config.NumberColumn("増減", step=2, format="%+d"),
                "AI予測_複勝確率": st.column_config.NumberColumn("AI複勝率", format="%.1f%%", disabled=True),
                "調教スコア": st.column_config.NumberColumn("調教", format="%.1f", disabled=True), "総合AIスコア": st.column_config.NumberColumn("総合AI", format="%.1f", disabled=True),
                "馬番複勝率": st.column_config.NumberColumn("馬番複勝率", format="%.1f%%", disabled=True),
            },
            hide_index=True, use_container_width=True, key=f"editor_{selected_venue}_{int(selected_race)}"
        )

        data_changed = False
        for i, row in edited.iterrows():
            m_idx = st.session_state.master_data[st.session_state.master_data['馬名'] == row['馬名']].index
            if not m_idx.empty:
                w_master, w_edited = st.session_state.master_data.loc[m_idx[0], '馬体重'], row['馬体重']
                z_master, z_edited = st.session_state.master_data.loc[m_idx[0], '増減'], row['増減']
                changed_w = (w_master != w_edited) and not (pd.isna(w_master) and pd.isna(w_edited))
                changed_z = (z_master != z_edited) and not (pd.isna(z_master) and pd.isna(z_edited))
                if changed_w or changed_z:
                    st.session_state.master_data.loc[m_idx, '馬体重'], st.session_state.master_data.loc[m_idx, '増減'] = w_edited, z_edited
                    data_changed = True
                
        if data_changed: save_backup(); st.rerun()  

        st.markdown("---")
        st.markdown("#### ⚖️ 馬体重入力")
        mc1, mc2, mc3, mc4 = st.columns([3, 2, 2, 2])
        with mc1:
            horse_options = race_df['馬番'].astype(str) + "番: " + race_df['馬名']
            selected_horse_str = st.selectbox("🐴 馬を選択", horse_options, key="m_horse")
            target_horse = selected_horse_str.split(": ")[1] if selected_horse_str else ""

        if target_horse:
            curr_w = st.session_state.master_data.loc[st.session_state.master_data['馬名'] == target_horse, '馬体重'].values[0]
            curr_z = st.session_state.master_data.loc[st.session_state.master_data['馬名'] == target_horse, '増減'].values[0]
            val_w = int(curr_w) if pd.notna(curr_w) and curr_w != 0 else 480
            val_z = int(curr_z) if pd.notna(curr_z) else 0

            with mc2: new_w = st.number_input("⚖️ 馬体重", value=val_w, step=2, key="m_w")
            with mc3: new_z = st.number_input("📈 増減", value=val_z, step=2, key="m_z")
            with mc4:
                st.write("") ; st.write("")
                if st.button("💾 馬体重反映", type="secondary", use_container_width=True, key="m_btn"):
                    m_idx = st.session_state.master_data[st.session_state.master_data['馬名'] == target_horse].index
                    st.session_state.master_data.loc[m_idx, '馬体重'], st.session_state.master_data.loc[m_idx, '増減'] = new_w, new_z
                    save_backup()
                    st.session_state.weight_updated_just_now = target_horse
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⚡ EXECUTE TRIPLE-AI ENGINE", type="primary", use_container_width=True):
            with st.spinner('🔄 階層型メタAIで予測スコアを算出中...'):
                time.sleep(0.6) 
                st.session_state.current_result = run_analysis(st.session_state.master_data)
                save_backup()
                st.session_state.predicted_just_now = True
            st.rerun()

# --- 成績分析タブ ---
with tab2:
    st.header("📊 PERFORMANCE (成績分析)")
    st.info("💡 予想を実行した後に、その日の結果CSVをアップロードすると回収率や勝率が表示されます。")
    res_file = st.file_uploader("📂 UPLOAD RESULTS (結果CSV)", type=["csv"], key=f"res_{st.session_state.file_key}")
    
    if res_file and st.session_state.get('current_result') is not None:
        try:
            df_res = read_uploaded_file(res_file)
            df_pred = st.session_state['current_result'].copy()
            def clean_pay(val):
                if pd.isna(val) or val == "": return 0
                s = str(val).replace(',', '').replace('(', '').replace(')', '').strip()
                try: return int(float(s))
                except: return 0
                
            df_res['単勝配当'] = df_res['単勝配当'].apply(clean_pay)
            df_res['複勝配当'] = df_res['複勝配導'].apply(clean_pay) if '複勝配導' in df_res.columns else df_res['複勝配当'].apply(clean_pay)
            df_res['確定着順'] = pd.to_numeric(df_res['確定着順'], errors='coerce')
            df_res['馬名'] = df_res['馬名'].astype(str).str.strip().apply(lambda x: unicodedata.normalize('NFKC', x))
            df_res = df_res.drop_duplicates(subset=['馬名'], keep='first')
            
            df_merge = pd.merge(df_pred, df_res[['馬名', '確定着順', '単勝配当', '複勝配当']], on='馬名', how='inner')
            match_ratio = len(df_merge) / len(df_pred) if len(df_pred) > 0 else 0
            
            if match_ratio < 0.5: st.error(f"⚠️ エラー：別の日のデータです。（一致した馬: {len(df_merge)}/{len(df_pred)}）")
            elif df_merge.empty: st.warning("⚠️ 結果データに一致する馬が見つかりませんでした。")
            else:
                marks = ['◎', '〇', '▲', '△']
                analysis = []
                for m in marks:
                    m_df = df_merge[df_merge['予想印'] == m]
                    if len(m_df) > 0:
                        win_r, pla_r = (m_df['確定着順'] == 1).mean() * 100, (m_df['確定着順'] <= 3).mean() * 100
                        win_roi, pla_roi = (m_df['単勝配当'].sum() / (len(m_df) * 100)) * 100, (m_df['複勝配当'].sum() / (len(m_df) * 100)) * 100
                        analysis.append({'印': m, '勝率': win_r, '複勝率': pla_r, '単勝回収': win_roi, '複勝回収': pla_roi, '対象頭数': len(m_df)})
                        
                df_ana = pd.DataFrame(analysis)
                cols = st.columns(len(analysis))
                for i, row in df_ana.iterrows(): cols[i].metric(f"印 {row['印']}", f"回収 {row['単勝回収']:.0f}%", f"勝率 {row['勝率']:.1f}%")
                st.table(df_ana.style.format({'勝率': '{:.1f}%', '複勝率': '{:.1f}%', '単勝回収': '{:.1f}%', '複勝回収': '{:.1f}%', '対象頭数': '{:d}頭'}))
                
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(px.bar(df_ana, x='印', y='単勝回収', title="単勝回収率 (%)", color='印', color_discrete_sequence=['#ff2800','#ffca28','#343a40','#0077b6']), use_container_width=True)
                with c2: st.plotly_chart(px.bar(df_ana, x='印', y='複勝率', title="複勝的中率 (%)", color='印', color_discrete_sequence=['#ff2800','#ffca28','#343a40','#0077b6']), use_container_width=True)
                
        except Exception as e: st.error(f"分析エラー: {e}")

# --- MANUALタブ ---
with tab3:
    st.header("📖 MANUAL (取扱説明書 & システム仕様)")
    st.markdown("""
    ### 🏁 SYSTEM WORKFLOW (使い方)
    本システムは、過去5年分の膨大なレースデータと、直前のコンディションを掛け合わせて最強の予測を出力します。

    1. **📥 データのセットアップ (SIDEBAR)**
       画面左側のメニューから「出馬表 (CSV)」と「調教データ (CSV)」をアップロードし、『💾 ファイルをシステムに記憶させる』を押します。
    2. **🎯 AIチューニング (SIDEBAR)**
       『🎯 Optunaチューニング実行』を押すと、AIが自己学習を行い、あなたのデータに最適なパラメータ（設定）を導き出します。
    3. **📍 レースと環境の指定**
       「競馬場」「レース番号」「天候」「馬場状態」を指定します。
    4. **🎛️ 本日の馬場バイアス補正**
       当日、実際のレースを見て「内枠有利だな」と感じたら、スライダーを動かすことでAIの最終スコアにボーナス点を加算させることができます。
    5. **⚡ AI予測の実行**
       『⚡ EXECUTE TRIPLE-AI ENGINE』ボタンを押すと、階層型メタAIが並列処理を開始し、「総合AIスコア」と「予想印（◎〇▲△）」を弾き出します。

    ---

    ### 🧠 META-AI STACKING ENGINE (搭載モデルと役割)
    本システムは、単一のAIではなく**「アンサンブル学習（スタッキング手法）」**を採用しています。得意分野の違う3つの最先端機械学習モデルが独自の視点で確率を算出し、それを第4のAI（メタモデル）が統合して最終判断を下します。

    * **🔵 LightGBM:** 処理速度と精度のバランスが優れた王道AI。
    * **🔴 XGBoost:** 荒れ馬場や大穴の激走を検知するアノマリー特化AI。
    * **🟡 CatBoost:** 血統や黄金コンビなど文字列データに強いスペシャリストAI。
    * **👑 Meta AI (Logistic Regression):** 上記3つのAIの「どれを信用すべきか」を学習し、比重を自動最適化する総司令官。

    ---

    ### ⚙️ PREDICTION LOGIC & FEATURES (予測ロジックと全入力ファクター)
    プロの馬券師が頭の中で行っている複雑な計算をすべて数値化し、**合計18種類以上のファクター**としてAIに学習させています。

    **【1】 基礎能力 ＆ 適性パラメーター**
    * **⏱️ 過去平均上り3F:** 展開が向いた時に確実に差し切る能力。
    * **🧬 場所別血統複勝率:** 該当競馬場における種牡馬・母父の勝率。
    * **🔢 コース別馬番複勝率:** 枠順の有利不利。

    **【2】 陣営の本気度 ＆ 人間関係**
    * **🏇 場所別騎手複勝率:** 該当競馬場におけるジョッキーの得意・不得意。
    * **🤝 黄金コンビ勝率:** 調教師×騎手の勝負気配。
    * **🔄 継続騎乗フラグ:** 前走と同じジョッキーが連続して騎乗するか。

    **【3】 物理的・展開的ファクター**
    * **🛡️ 斤量体重比:** 「背負う斤量 ÷ 馬体重」によるパワー指標。
    * **📏 距離増減:** 前走の距離との差分（短縮・延長）の適性。
    * **🔥 逃げ馬頭数:** 出走馬の中の「前走逃げた馬」の数からハイペース/スローペースを検知。

    **【4】 リアルタイム・チューニング**
    * **🏃‍♂️ 調教スコア:** コースごとに偏差値（Tスコア）化した直前の仕上がり。
    * **⚖️ 馬体重増減:** 発表された当日の馬体重による手動補正。
    * **🎛️ 馬場バイアス:** ユーザーの目視による当日の「前残り・内枠有利」などのトラックバイアス補正。
    """)