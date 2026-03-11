import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import unicodedata
import warnings
import os
import base64
import plotly.express as px

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI HORSE RACING SYSTEM", layout="wide")

# --- 🎨 デザインCSS ---
st.markdown("""
    <style>
        .block-container { padding-top: 1.2rem !important; }
        .stDataFrame { border: 1px solid #e63946; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 🏎️ ヘッダー ---
img_name = "ferrari.png" 
BANNER_RED, BANNER_YELLOW, TEXT_COLOR = "#ff2800", "#ffca28", "#000000"

if os.path.exists(img_name):
    with open(img_name, "rb") as f: img_base64 = base64.b64encode(f.read()).decode()
    img_html = f"<img src='data:image/png;base64,{img_base64}' style='width: 100%; max-width: 380px; object-fit: contain;'>"
else:
    img_html = f"<h1 style='font-size: 8rem; margin:0; color: {TEXT_COLOR};'>🏇</h1>"

banner_html = f"""
<div style="background-color: {BANNER_RED}; padding: 15px; border-radius: 15px; box-shadow: 0 8px 20px rgba(0,0,0,0.3); margin-bottom: 20px;">
    <div style="background-color: {BANNER_YELLOW}; border-radius: 10px; padding: 20px 30px; display: flex; align-items: center;">
        <div style="flex: 1; text-align: center;">{img_html}</div>
        <div style="flex: 2; padding-left: 30px;">
            <h1 translate='no' class='notranslate' style='color: {TEXT_COLOR}; font-family: Arial, sans-serif; font-size: 3.2rem; font-weight: 900; margin: 0;'>AI HORSE RACING SYSTEM</h1>
            <p style='color: {TEXT_COLOR}; margin-top: 10px; font-size: 1.1rem; font-weight: bold;'>ADVANCED PREDICTION & REAL-TIME ANALYSIS</p>
        </div>
    </div>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

# --- 🧠 AI学習エンジン ---
@st.cache_data
def load_and_train_ai():
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
    # 「芝・ダ」または「芝ダ」または「芝田」を統一して読み込む
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
    surface_map = {'芝':1, 'ダ':2, '田':2, '障':3} # 念のため「田」もダートとして登録
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

    return model, j_stats, s_stats, g_stats, t_stats, horse_agg, place_map, weather_map, track_map, surface_map, train_features, style_map, df_past

def read_uploaded_file(uploaded_file, is_syutuba=False):
    kwargs = {'engine': 'python'}
    if is_syutuba: kwargs['header'] = None
    else: kwargs['on_bad_lines'] = 'skip'
    try: df = pd.read_csv(uploaded_file, encoding='utf-8', **kwargs)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='cp932', **kwargs)
    return df

def calc_dev(series):
    s_clean = series.dropna()
    if len(s_clean) < 2: return pd.Series([np.nan]*len(series), index=series.index)
    return -(series - s_clean.mean()) / (s_clean.std() + 0.001) * 10 + 50

def get_waku_color(val):
    try: val = int(val)
    except: return ''
    colors = {1:'#f8f9fa', 2:'#343a40', 3:'#e63946', 4:'#0077b6', 5:'#ffd166', 6:'#2a9d8f', 7:'#f77f00', 8:'#ffb5a7'}
    return colors.get(val, '')

# 初期化
model, j_stats, s_stats, g_stats, t_stats, horse_agg, place_map, weather_map, track_map, surface_map, train_features, style_map, df_full_past = load_and_train_ai()

if 'last_predictions' not in st.session_state: st.session_state['last_predictions'] = None

# --- 📱 タブ構成 ---
tab1, tab2 = st.tabs(["🔮 AI PREDICTION (予想)", "📊 PERFORMANCE (成績分析)"])

with tab1:
    st.sidebar.header("⚙️ SETTINGS")
    weather_setting = st.sidebar.selectbox("WEATHER", ["指定なし", "晴", "曇", "小雨", "雨", "小雪", "雪"])
    track_setting = st.sidebar.selectbox("TRACK", ["指定なし", "良", "稍重", "重", "不良"])
    st.sidebar.markdown("---")
    syutuba_file = st.sidebar.file_uploader("📂 RACE CARD (24項目CSV)", type=["csv"])
    training_files = st.sidebar.file_uploader("📂 TRAINING DATA", type=["csv"], accept_multiple_files=True)
    run_button = st.sidebar.button("⚡ EXECUTE AI ANALYSIS", use_container_width=True)

    if run_button and syutuba_file:
        with st.spinner("AI is analyzing..."):
            try:
                df_raw = read_uploaded_file(syutuba_file, is_syutuba=True)
                df_all = df_raw.iloc[:, :24].copy()
                df_all.columns = ['枠番', '馬番', '場所', 'R', 'レース名', '芝ダ', '距離', '頭数', '馬名', '時刻', '条件', 'B', '性別', '年齢', '騎手', '斤量', '馬体重', '増減', '所属', '調教師', '父', '母父', '前走着順', '脚質']
                
                # 💥 「クラックス」「ｸﾗｽ」を徹底修正
                df_all['レース名'] = df_all['レース名'].astype(str).str.replace('クラッス', 'クラス').str.replace('クラックス', 'クラス').str.replace('ｸﾗｽ', 'クラス')
                df_all['レース名'] = df_all['レース名'].apply(lambda x: unicodedata.normalize('NFKC', x))
                
                # 「芝ダ」の表記揺れ（芝田）対策
                df_all['芝ダ'] = df_all['芝ダ'].astype(str).str.strip().str.replace('田', 'ダ')
                
                df_all['馬番'] = pd.to_numeric(df_all['馬番'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
                df_all['枠番'] = pd.to_numeric(df_all['枠番'], errors='coerce')
                df_all['馬名'] = df_all['馬名'].astype(str).str.strip()
                df_all['脚質_num'] = df_all['脚質'].astype(str).str.strip().map(style_map).fillna(3)
                df_all['前走確定着順'] = pd.to_numeric(df_all['前走着順'], errors='coerce').fillna(10)
                df_all['斤量'] = pd.to_numeric(df_all['斤量'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
                df_all['馬体重'] = pd.to_numeric(df_all['馬体重'], errors='coerce')

                # スタッツ結合（マージのキーを確実に合わせる）
                df_all = pd.merge(df_all, j_stats, on=['場所', '騎手'], how='left')
                df_all = pd.merge(df_all, s_stats, left_on=['場所', '父'], right_on=['場所', '種牡馬'], how='left')
                df_all = pd.merge(df_all, g_stats, left_on=['場所', '芝ダ', '馬番'], right_on=['場所', '芝ダ_str', '馬番'], how='left')
                if not t_stats.empty: df_all = pd.merge(df_all, t_stats, on=['場所', '調教師'], how='left')
                df_all = pd.merge(df_all, horse_agg, on='馬名', how='left')

                df_all['天候_num'] = weather_map.get(weather_setting, np.nan)
                df_all['馬場状態_num'] = track_map.get(track_setting, np.nan)
                df_all['芝ダ_num'] = df_all['芝ダ'].map(surface_map)
                df_all['場所_num'] = df_all['場所'].map(place_map)
                df_all['距離'] = pd.to_numeric(df_all['距離'], errors='coerce')
                df_all['年齢'] = pd.to_numeric(df_all['年齢'], errors='coerce')

                # AI予測
                df_all['AI予測_複勝確率'] = (model.predict_proba(df_all[train_features].fillna(0))[:, 1] * 100).round(1)

                # 調教スコア
                df_all['調教スコア'] = 50.0 
                if training_files:
                    df_train_list = [read_uploaded_file(f) for f in training_files]
                    df_training = pd.concat(df_train_list, ignore_index=True)
                    if '馬名' in df_training.columns and 'Lap1' in df_training.columns:
                        df_training['馬名'] = df_training['馬名'].astype(str).str.strip()
                        df_training['Lap1'] = pd.to_numeric(df_training['Lap1'], errors='coerce')
                        df_training['Lap2'] = pd.to_numeric(df_training['Lap2'], errors='coerce')
                        df_training['加速'] = df_training['Lap1'] - df_training['Lap2']
                        train_agg = df_training.groupby('馬名').agg(L1=('Lap1','mean'), K=('加速','mean')).reset_index()
                        df_all = pd.merge(df_all, train_agg, on='馬名', how='left')
                        df_all['調教スコア'] = (calc_dev(df_all['L1']) + calc_dev(df_all['K'])) / 2
                
                df_all['調教スコア'] = df_all['調教スコア'].fillna(50).round(1)
                df_all['総合AIスコア'] = (df_all['AI予測_複勝確率'] * 0.75) + (df_all['調教スコア'] * 0.25)
                df_all['総合AIスコア'] = df_all['総合AIスコア'].round(1)

                # 印付け
                final_list = []
                for _, group in df_all.groupby(['場所', 'R', 'レース名']):
                    group = group.sort_values('総合AIスコア', ascending=False).reset_index(drop=True)
                    group['予想印'] = [('◎' if i==0 else '〇' if i==1 else '▲' if i==2 else '△' if i<=5 else '') for i in range(len(group))]
                    final_list.append(group)
                df_final = pd.concat(final_list)
                st.session_state['last_predictions'] = df_final

                for r_id, group in df_final.groupby(['場所', 'R', 'レース名']):
                    st.markdown(f"### 🏁 {r_id[0]} {r_id[1]}R - {r_id[2]}")
                    
                    # 💥 表示項目をすべて復活！
                    disp = group[['予想印', '馬番', '馬名', '斤量', '騎手', 'コース別馬番複勝率', 'AI予測_複勝確率', '調教スコア', '総合AIスコア', '脚質']].copy()
                    
                    def format_weight(row):
                        idx = row.name
                        w, c = group.loc[idx, '馬体重'], group.loc[idx, '増減']
                        if pd.isna(w) or str(w).strip() == "nan": return "未発表"
                        return f"{int(w)}({str(c)})" if not pd.isna(c) and str(c).strip() != "nan" else f"{int(w)}"
                            
                    disp.insert(3, '馬体重(増減)', group.apply(format_weight, axis=1))
                    disp['コース別馬番複勝率'] = (disp['コース別馬番複勝率']*100).round(1).astype(str)+"%"
                    disp['AI予測_複勝確率'] = disp['AI予測_複勝確率'].astype(str)+"%"
                    
                    def apply_waku(row):
                        w = int(group.iloc[row.name]['枠番'])
                        c = get_waku_color(w)
                        txt = 'white' if w in [2,3,4,6,7] else 'black'
                        return [f'background-color: {c}; color: {txt}' if col=='馬番' else '' for col in row.index]
                    
                    st.dataframe(disp.style.apply(apply_waku, axis=1), hide_index=True, use_container_width=True)
                    st.markdown("---")
            except Exception as e: st.error(f"予測エラー: {e}")

with tab2:
    st.header("📊 PERFORMANCE (成績分析)")
    st.write("今回の予想印と実際の結果を照らし合わせて、的中率と回収率を算出します。")
    
    res_file = st.file_uploader("📂 UPLOAD RESULTS (結果CSV)", type=["csv"])
    
    if res_file and st.session_state['last_predictions'] is not None:
        try:
            df_res = read_uploaded_file(res_file)
            df_pred = st.session_state['last_predictions']
            
            def clean_pay(val):
                if pd.isna(val) or val == "": return 0
                s = str(val).replace(',', '').replace('(', '').replace(')', '').strip()
                try: return int(float(s))
                except: return 0

            df_res['単勝配当'] = df_res['単勝配当'].apply(clean_pay)
            df_res['複勝配当'] = df_res['複勝配当'].apply(clean_pay)
            df_res['確定着順'] = pd.to_numeric(df_res['確定着順'], errors='coerce')
            df_res['馬名'] = df_res['馬名'].astype(str).str.strip()

            df_merge = pd.merge(df_pred, df_res[['馬名', '確定着順', '単勝配当', '複勝配当']], on='馬名', how='inner')
            
            if not df_merge.empty:
                # 💥 ラベルを日本語で分かりやすく修正
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
                
                # サマリー
                cols = st.columns(len(analysis))
                for i, row in df_ana.iterrows():
                    cols[i].metric(f"印 {row['印']}", f"回収 {row['単勝回収']:.0f}%", f"勝率 {row['勝率']:.1f}%")

                st.markdown("### 📈 分析データ")
                st.table(df_ana.style.format({
                    '勝率': '{:.1f}%', '複勝率': '{:.1f}%', 
                    '単勝回収': '{:.1f}%', '複勝回収': '{:.1f}%', '対象頭数': '{:d}頭'
                }))

                # グラフ
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.bar(df_ana, x='印', y='単勝回収', title="印ごとの単勝回収率 (%)", color='印', color_discrete_sequence=['#ff2800','#ffca28','#343a40','#0077b6']), use_container_width=True)
                with c2:
                    st.plotly_chart(px.bar(df_ana, x='印', y='複勝率', title="印ごとの複勝的中率 (%)", color='印', color_discrete_sequence=['#ff2800','#ffca28','#343a40','#0077b6']), use_container_width=True)
            else:
                st.warning("馬名が一致するデータが見つかりませんでした。")
        except Exception as e: st.error(f"分析エラー: {e}")