import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import unicodedata
import warnings
import os
import base64
import plotly.express as px
import pickle
import re

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI HORSE RACING SYSTEM", layout="wide")

# --- 💾 超強力：自動バックアップ＆復元機能（エラー防止強化版） ---
def load_backup():
    try:
        if 'master_data' not in st.session_state and os.path.exists('backup_master.pkl'):
            st.session_state.master_data = pd.read_pickle('backup_master.pkl')
        if 'training_df' not in st.session_state and os.path.exists('backup_training.pkl'):
            st.session_state.training_df = pd.read_pickle('backup_training.pkl')
        if 'current_result' not in st.session_state and os.path.exists('backup_result.pkl'):
            st.session_state.current_result = pd.read_pickle('backup_result.pkl')
    except Exception:
        pass 

def save_backup():
    try:
        if 'master_data' in st.session_state:
            st.session_state.master_data.to_pickle('backup_master.pkl')
        if 'training_df' in st.session_state:
            st.session_state.training_df.to_pickle('backup_training.pkl')
        if 'current_result' in st.session_state:
            st.session_state.current_result.to_pickle('backup_result.pkl')
    except PermissionError:
        pass 
    except Exception:
        pass

# アプリ起動時にバックアップがあれば自動で読み込む
load_backup()

# --- 🎨 デザインCSS & ヘッダー ---
img_name = "ferrari.png" 
BANNER_RED, BANNER_YELLOW, TEXT_COLOR = "#ff2800", "#ffca28", "#000000"

st.markdown("""
<style>
@keyframes pulse-box {
    0% { transform: scale(1); box-shadow: 0 0 10px rgba(255, 40, 0, 0.7); }
    50% { transform: scale(1.03); box-shadow: 0 0 25px rgba(255, 202, 40, 1); }
    100% { transform: scale(1); box-shadow: 0 0 10px rgba(255, 40, 0, 0.7); }
}
.glow-guide {
    animation: pulse-box 1.2s infinite ease-in-out;
    background: linear-gradient(135deg, #ff2800, #ffca28);
    color: #ffffff !important;
    font-size: 1.05rem;
    font-weight: 900;
    text-align: center;
    padding: 12px;
    border-radius: 8px;
    margin-top: 15px;
    margin-bottom: 5px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
}
.streamlit-expanderHeader {
    font-weight: bold;
    color: #333 !important;
    background-color: #f8f9fa;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

def get_banner():
    if os.path.exists(img_name):
        with open(img_name, "rb") as f: img_base64 = base64.b64encode(f.read()).decode()
        img_html = f"<img src='data:image/png;base64,{img_base64}' style='width: 100%; max-width: 380px; object-fit: contain;'>"
    else:
        img_html = f"<h1 style='font-size: 8rem; margin:0;'>🏇</h1>"
    return f"""
    <div style="background-color: {BANNER_RED}; padding: 15px; border-radius: 15px; margin-bottom: 20px;">
        <div style="background-color: {BANNER_YELLOW}; border-radius: 10px; padding: 20px 30px; display: flex; align-items: center;">
            <div style="flex: 1; text-align: center;">{img_html}</div>
            <div style="flex: 2; padding-left: 30px;">
                <h1 translate='no' class='notranslate' style='color: {TEXT_COLOR}; font-family: Arial; font-size: 3.2rem; font-weight: 900; margin: 0;'>AI HORSE RACING SYSTEM</h1>
                <p style='color: {TEXT_COLOR}; font-size: 1.1rem; font-weight: bold;'>PREDICTION & REAL-TIME EDIT</p>
            </div>
        </div>
    </div>"""

st.markdown(get_banner(), unsafe_allow_html=True)

# --- 🧠 AI学習エンジン ---
@st.cache_resource
def load_and_train_ai():
    if os.path.exists('5yers_data.zip'):
        file_path = '5yers_data.zip'
    else:
        file_path = '5yers_data.csv'

    try:
        df_past = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
    except:
        df_past = pd.read_csv(file_path, encoding='cp932', on_bad_lines='skip', engine='python')

    df_past = df_past.rename(columns={'着順': '確定着順', '天気': '天候'})
    
    def clean_rank(x):
        s = unicodedata.normalize('NFKC', str(x))
        nums = "".join([c for c in s if c.isdigit()])
        return int(nums) if nums else 99
    
    df_past['確定着順_num'] = df_past['確定着順'].apply(clean_rank)
    df_past['is_top3'] = df_past['確定着順_num'].apply(lambda x: 1 if x <= 3 else 0)

    style_map = {'逃げ': 1, '先行': 2, '中団': 3, '後方': 4, 'ﾏｸﾘ': 5}
    df_past['脚質_num'] = df_past['脚質'].astype(str).str.strip().map(style_map).fillna(3)

    df_past['馬番'] = pd.to_numeric(df_past['馬番'], errors='coerce')
    df_past['芝ダ_str'] = df_past['芝・ダ'].astype(str).str.strip().str.replace('田', 'ダ')
    
    j_stats = df_past.groupby(['場所', '騎手'])['is_top3'].mean().reset_index().rename(columns={'is_top3': '場所別騎手複勝率'})
    s_stats = df_past.groupby(['場所', '種牡馬'])['is_top3'].mean().reset_index().rename(columns={'is_top3': '場所別血統複勝率'})
    g_stats = df_past.groupby(['場所', '芝ダ_str', '馬番'])['is_top3'].mean().reset_index().rename(columns={'is_top3': 'コース別馬番複勝率'})
    t_stats = df_past.groupby(['場所', '調教師'])['is_top3'].mean().reset_index().rename(columns={'is_top3': '場所別調教師複勝率'}) if '調教師' in df_past.columns else pd.DataFrame()

    df_past = pd.merge(df_past, j_stats, on=['場所', '騎手'], how='left')
    df_past = pd.merge(df_past, s_stats, on=['場所', '種牡馬'], how='left')
    df_past = pd.merge(df_past, g_stats, on=['場所', '芝ダ_str', '馬番'], how='left')
    if not t_stats.empty: df_past = pd.merge(df_past, t_stats, on=['場所', '調教師'], how='left')

    horse_agg = df_past.groupby('馬名').agg(過去平均上り3F=('上り3F', 'mean')).reset_index()
    df_past = pd.merge(df_past, horse_agg, on='馬名', how='left')

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
                      '脚質_num', '前走確定着順']
    
    for col in train_features: df_past[col] = pd.to_numeric(df_past[col], errors='coerce').fillna(0)
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, learning_rate=0.1, verbose=-1)
    model.fit(df_past[train_features], df_past['is_top3'])

    return model, j_stats, s_stats, g_stats, t_stats, horse_agg, place_map, weather_map, track_map, surface_map, train_features, style_map

# --- ヘルパー関数 ---
def read_uploaded_file(uploaded_file, is_syutuba=False):
    uploaded_file.seek(0)
    kwargs = {'engine': 'python'}
    if is_syutuba: kwargs['header'] = None
    else: kwargs['on_bad_lines'] = 'skip'
    try: 
        df = pd.read_csv(uploaded_file, encoding='utf-8', **kwargs)
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

model, j_stats, s_stats, g_stats, t_stats, horse_agg, place_map, weather_map, track_map, surface_map, train_features, style_map = load_and_train_ai()

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ データ読み込み")
    
    is_master_ready = 'master_data' in st.session_state
    is_result_ready = 'current_result' in st.session_state
    
    syutuba_file_raw = st.session_state.get("syutuba_uploader_key")
    syutuba_uploaded = bool(syutuba_file_raw)

    if not is_master_ready and not syutuba_uploaded:
        st.markdown('<div class="glow-guide">👇【STEP 1】まずは「出馬表」をココに！</div>', unsafe_allow_html=True)
    
    syutuba_file = st.file_uploader("📂 出馬表 (CSV)", type=["csv"], key="syutuba_uploader_key")

    if not is_master_ready and syutuba_uploaded:
        st.markdown('<div class="glow-guide">👇【STEP 2】調教データがあればココに！</div>', unsafe_allow_html=True)
        
    training_files = st.file_uploader("📂 調教データ (CSV)", type=["csv"], accept_multiple_files=True, key="training_uploader_key")
    
    if not is_master_ready and syutuba_uploaded:
        st.markdown('<div class="glow-guide">👇【STEP 3】ココをタップして記憶させる！</div>', unsafe_allow_html=True)

    save_btn_type = "primary" if (not is_master_ready and syutuba_uploaded) else "secondary"
    save_clicked = st.button("💾 ファイルをシステムに記憶させる", type=save_btn_type, use_container_width=True)
    
    if save_clicked:
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
            st.session_state.last_filename = syutuba_file.name
            save_backup()
            st.success("✅ 出馬表を記憶しました！")
            
        if training_files:
            dfs_t = [read_uploaded_file(f) for f in training_files]
            df_training = pd.concat(dfs_t, ignore_index=True)
            st.session_state.training_df = df_training
            save_backup()
            st.success("✅ 調教データを記憶しました！")
            
        st.rerun()

    st.markdown("---")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("🗑️ オールクリア (全データ初期化)", type="secondary", use_container_width=True):
        st.session_state.clear()
        for f in ['backup_master.pkl', 'backup_training.pkl', 'backup_result.pkl']:
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
    
    df_work = pd.merge(df_work, j_stats, on=['場所', '騎手'], how='left')
    df_work = pd.merge(df_work, s_stats, left_on=['場所', '父'], right_on=['場所', '種牡馬'], how='left')
    df_work = pd.merge(df_work, g_stats, left_on=['場所', '芝ダ', '馬番'], right_on=['場所', '芝ダ_str', '馬番'], how='left')
    if not t_stats.empty: df_work = pd.merge(df_work, t_stats, on=['場所', '調教師'], how='left')
    df_work = pd.merge(df_work, horse_agg, on='馬名', how='left')

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
    df_work['AI予測_複勝確率'] = (model.predict_proba(X)[:, 1] * 100).round(1)

    df_work['調教スコア'] = 50.0
    if 'training_df' in st.session_state:
        df_training = st.session_state.training_df.copy()
        if '馬名' in df_training.columns and 'Lap1' in df_training.columns:
            df_training['馬名'] = df_training['馬名'].astype(str).str.strip()
            df_training['Lap1'] = pd.to_numeric(df_training['Lap1'], errors='coerce')
            df_training['Lap2'] = pd.to_numeric(df_training['Lap2'], errors='coerce')
            df_training['加速'] = df_training['Lap1'] - df_training['Lap2']
            
            course_col = None
            if '調教種別' in df_training.columns: course_col = '調教種別'
            elif 'コース' in df_training.columns: course_col = 'コース'

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
    df_work['総合AIスコア'] = ((df_work['AI予測_複勝確率'] * 0.75) + (df_work['調教スコア'] * 0.25)).round(1)
    df_work['コース別馬番複勝率_disp'] = (df_work['コース別馬番複勝率'] * 100).round(1)
    
    final = []
    for _, g in df_work.groupby(['場所', 'R', 'レース名']):
        g = g.sort_values('総合AIスコア', ascending=False).reset_index(drop=True)
        g['予想印'] = [('◎' if i==0 else '〇' if i==1 else '▲' if i==2 else '△' if i<=5 else '') for i in range(len(g))]
        final.append(g)
    return pd.concat(final)


# --- メイン画面 (タブ) ---
tab1, tab2 = st.tabs(["🔮 AI PREDICTION (予想)", "📊 PERFORMANCE (成績分析)"])

with tab1:
    if 'master_data' in st.session_state:
        st.markdown("### 🏁 レース選択 & 条件設定")
        
        c1, c2 = st.columns([1.5, 1.5])
        st.session_state.master_data['R'] = pd.to_numeric(st.session_state.master_data['R'], errors='coerce').fillna(0).astype(int)
        venues = st.session_state.master_data['場所'].unique()
        
        with c1:
            selected_venue = st.selectbox("📍 競馬場", venues, key="sel_venue")
        
        races = st.session_state.master_data[st.session_state.master_data['場所'] == selected_venue]['R'].unique()
        with c2:
            selected_race = st.selectbox("🏁 R", sorted(races), key="sel_race")

        with st.expander("⛅ 天候・馬場状態の設定 (タップで開閉)", expanded=False):
            wc1, wc2, wc3 = st.columns(3)
            with wc1:
                w_val = st.selectbox("天候", ["指定なし", "晴", "曇", "小雨", "雨", "小雪", "雪"], key=f"w_{selected_venue}")
                st.session_state[f"weather_{selected_venue}"] = w_val
            with wc2:
                t_shiba_val = st.selectbox("芝", ["指定なし", "良", "稍重", "重", "不良"], key=f"ts_{selected_venue}")
                st.session_state[f"track_shiba_{selected_venue}"] = t_shiba_val
            with wc3:
                t_dirt_val = st.selectbox("ダート", ["指定なし", "良", "稍重", "重", "不良"], key=f"td_{selected_venue}")
                st.session_state[f"track_dirt_{selected_venue}"] = t_dirt_val

        st.markdown("---")

        race_df = st.session_state.master_data[
            (st.session_state.master_data['場所'] == selected_venue) & 
            (st.session_state.master_data['R'] == selected_race)
        ].copy()
        
        race_df['馬番'] = pd.to_numeric(race_df['馬番'], errors='coerce')
        race_df = race_df.sort_values('馬番').reset_index(drop=True)
        
        raw_race_name = str(race_df['レース名'].iloc[0]) if not race_df.empty else ""
        clean_name = re.sub(r'第?\d+回', '', raw_race_name)
        clean_name = re.sub(r'\d+日目?', '', clean_name)
        clean_name = re.sub(r'[^\s]*競馬場', '', clean_name)
        clean_name = re.sub(r'\d+[RＲrｒ]', '', clean_name)
        clean_name = clean_name.replace(selected_venue, '')
        clean_name = re.sub(r'^[\s\-ー_]+', '', clean_name) 
        clean_name = clean_name.strip()

        if clean_name == "" or clean_name.lower() == "nan":
            st.subheader(f"🏆 {selected_venue} {int(selected_race)}R")
        else:
            st.subheader(f"🏆 {selected_venue} {int(selected_race)}R - {clean_name}")

        if 'current_result' in st.session_state:
            pred_df = st.session_state.current_result[
                (st.session_state.current_result['場所'] == selected_venue) & 
                (st.session_state.current_result['R'] == selected_race)
            ]
            if not pred_df.empty:
                merge_cols = ['馬名', '予想印', '総合AIスコア', 'AI予測_複勝確率', '調教スコア']
                if 'コース別馬番複勝率_disp' in pred_df.columns:
                    merge_cols.append('コース別馬番複勝率_disp')
                    
                race_df = pd.merge(race_df, pred_df[merge_cols], on='馬名', how='left')
                if 'コース別馬番複勝率_disp' in race_df.columns:
                    race_df = race_df.rename(columns={'コース別馬番複勝率_disp': '馬番複勝率'})
            else:
                race_df['予想印'] = ""
        else:
            race_df['予想印'] = ""

        disp_cols = ['予想印', '枠番', '馬番', '馬名', '斤量', '騎手', '馬体重', '増減']
        
        if 'AI予測_複勝確率' in race_df.columns:
            disp_cols.extend(['AI予測_複勝確率', '調教スコア', '総合AIスコア'])
        if '馬番複勝率' in race_df.columns:
            disp_cols.append('馬番複勝率')

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
                "予想印": st.column_config.Column("予想印", disabled=True),
                "枠番": None, 
                "馬番": st.column_config.NumberColumn("馬番", disabled=True),
                "馬名": st.column_config.Column("馬名", disabled=True),
                "斤量": st.column_config.NumberColumn("斤量", disabled=True),
                "騎手": st.column_config.Column("騎手", disabled=True),
                "馬体重": st.column_config.NumberColumn("馬体重", min_value=300, max_value=700, step=2),
                "増減": st.column_config.NumberColumn("増減", step=2, format="%+d"),
                "AI予測_複勝確率": st.column_config.NumberColumn("AI複勝率", format="%.1f%%", disabled=True),
                "調教スコア": st.column_config.NumberColumn("調教", format="%.1f", disabled=True),
                "総合AIスコア": st.column_config.NumberColumn("総合AI", format="%.1f", disabled=True),
                "馬番複勝率": st.column_config.NumberColumn("馬番複勝率", format="%.1f%%", disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key=f"editor_{selected_venue}_{int(selected_race)}"
        )

        # 🚀【魔法の修正】NaN同士の比較による無限ループバグを完全に阻止！
        data_changed = False
        for i, row in edited.iterrows():
            m_idx = st.session_state.master_data[st.session_state.master_data['馬名'] == row['馬名']].index
            
            if not m_idx.empty:
                w_master = st.session_state.master_data.loc[m_idx[0], '馬体重']
                w_edited = row['馬体重']
                z_master = st.session_state.master_data.loc[m_idx[0], '増減']
                z_edited = row['増減']
                
                # 「空っぽ(NaN)」と「空っぽ(NaN)」を比較してTrueにならないようにガード
                changed_w = (w_master != w_edited) and not (pd.isna(w_master) and pd.isna(w_edited))
                changed_z = (z_master != z_edited) and not (pd.isna(z_master) and pd.isna(z_edited))
                
                if changed_w or changed_z:
                    st.session_state.master_data.loc[m_idx, '馬体重'] = w_edited
                    st.session_state.master_data.loc[m_idx, '増減'] = z_edited
                    data_changed = True
                
        if data_changed:
            save_backup()
            st.rerun()  # 変更があった時のみ確実に1発リロード！

        st.markdown("---")

        # 📱 スマホ専用：馬体重サクサク入力ツール
        st.markdown("#### 📱 スマホ専用：馬体重サクサク入力ツール")
        st.info("💡 表のタップがやりにくい場合は、こちらを使うと「＋ / －」ボタンでサクサク入力できます！")
        
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

            with mc2:
                new_w = st.number_input("⚖️ 馬体重", value=val_w, step=2, key="m_w")
            with mc3:
                new_z = st.number_input("📈 増減", value=val_z, step=2, key="m_z")
            with mc4:
                st.write("") 
                st.write("")
                if st.button("💾 反映", type="secondary", use_container_width=True, key="m_btn"):
                    m_idx = st.session_state.master_data[st.session_state.master_data['馬名'] == target_horse].index
                    st.session_state.master_data.loc[m_idx, '馬体重'] = new_w
                    st.session_state.master_data.loc[m_idx, '増減'] = new_z
                    save_backup()
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # 無事復活した予想ボタン！
        if st.button("⚡ 予想を実行 / 更新する", type="primary", use_container_width=True):
            with st.spinner('AIが計算中...'):
                st.session_state.current_result = run_analysis(st.session_state.master_data)
                save_backup()
            st.rerun()


# --- 超・厳密マッチング機能付き：成績分析タブ ---
with tab2:
    st.header("📊 PERFORMANCE (成績分析)")
    st.info("💡 予想を実行した後に、その日の結果CSVをアップロードすると回収率や勝率が表示されます。")
    res_file = st.file_uploader("📂 UPLOAD RESULTS (結果CSV)", type=["csv"])
    
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
            
            if match_ratio < 0.5:
                st.error(f"⚠️ エラー：別の日のデータです。（予想した馬 {len(df_pred)} 頭のうち、結果データと一致したのは {len(df_merge)} 頭だけでした）")
            elif df_merge.empty:
                st.warning("⚠️ 結果データに一致する馬が見つかりませんでした。")
            else:
                marks = ['◎', '〇', '▲', '△']
                analysis = []
                for m in marks:
                    m_df = df_merge[df_merge['予想印'] == m]
                    if len(m_df) > 0:
                        win_r = (m_df['確定着順'] == 1).mean() * 100
                        pla_r = (m_df['確定着順'] <= 3).mean() * 100
                        win_roi = (m_df['単勝配当'].sum() / (len(m_df) * 100)) * 100
                        pla_roi = (m_df['複勝配当'].sum() / (len(m_df) * 100)) * 100
                        analysis.append({'印': m, '勝率': win_r, '複勝率': pla_r, '単勝回収': win_roi, '複勝回収': pla_roi, '対象頭数': len(m_df)})
                        
                df_ana = pd.DataFrame(analysis)
                
                cols = st.columns(len(analysis))
                for i, row in df_ana.iterrows():
                    cols[i].metric(f"印 {row['印']}", f"回収 {row['単勝回収']:.0f}%", f"勝率 {row['勝率']:.1f}%")
                    
                st.table(df_ana.style.format({'勝率': '{:.1f}%', '複勝率': '{:.1f}%', '単勝回収': '{:.1f}%', '複勝回収': '{:.1f}%', '対象頭数': '{:d}頭'}))
                
                c1, c2 = st.columns(2)
                with c1: 
                    st.plotly_chart(px.bar(df_ana, x='印', y='単勝回収', title="単勝回収率 (%)", color='印', color_discrete_sequence=['#ff2800','#ffca28','#343a40','#0077b6']), use_container_width=True)
                with c2: 
                    st.plotly_chart(px.bar(df_ana, x='印', y='複勝率', title="複勝的中率 (%)", color='印', color_discrete_sequence=['#ff2800','#ffca28','#343a40','#0077b6']), use_container_width=True)
                
        except Exception as e: 
            st.error(f"分析エラー: {e}")